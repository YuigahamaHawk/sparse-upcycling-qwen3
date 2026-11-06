"""Single-model inference HTTP server, with supervised GPU lease enforcement."""
from contextlib import asynccontextmanager
import logging
import secrets
import time
from typing import Literal
import uuid
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator
from models.inference.common import (BusyError, LeaseExpired, RequestError,
                            chat_prompt, instruction_prompt)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GenerateRequest(StrictRequest):
    model: str | None = None
    prompt: str | None = Field(default=None, min_length=1, max_length=100000)
    instruction: str | None = Field(default=None, min_length=1, max_length=32000)
    input: str = Field(default="", max_length=64000)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=42, ge=0, lt=2**32)
    stop: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def one_prompt(self):
        if (self.prompt is None) == (self.instruction is None):
            raise ValueError("provide exactly one of prompt or instruction")
        if self.prompt is not None and self.input:
            raise ValueError("input is allowed only with instruction")
        return self


class CompletionRequest(StrictRequest):
    model: str
    prompt: str = Field(min_length=1, max_length=100000)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=42, ge=0, lt=2**32)
    stop: str | list[str] | None = None
    stream: Literal[False] = False
    n: Literal[1] = 1


class ChatMessage(StrictRequest):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=64000)


class ChatRequest(StrictRequest):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=33)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=42, ge=0, lt=2**32)
    stop: str | list[str] | None = None
    stream: Literal[False] = False
    n: Literal[1] = 1


class LoadRequest(StrictRequest):
    model: str


def create_app(engine, initial_model=None, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        if initial_model:
            engine.load(initial_model)
        yield
        engine.close()

    app = FastAPI(title="Qwen3 sparse upcycling inference", version="1.0.1", lifespan=lifespan)
    app.state.engine = engine

    bearer = HTTPBearer(auto_error=False, scheme_name="ModelAPIKey",
                        description="Paste the API key only. Swagger adds the Bearer prefix automatically.")

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        # OpenAPI reserves Authorization: defining it as a plain Header makes
        # Swagger silently omit it. A bearer security scheme emits it correctly.
        if api_key and (credentials is None or not secrets.compare_digest(credentials.credentials, api_key)):
            raise HTTPException(status_code=401, detail="invalid bearer API key")

    @app.exception_handler(RequestError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=400, content={"error": {"type": "invalid_request_error", "message": str(exc)}})

    @app.exception_handler(BusyError)
    async def busy(request, exc):
        return JSONResponse(status_code=429, headers={"Retry-After": "5"},
                            content={"error": {"type": "model_busy", "message": str(exc)}})

    @app.exception_handler(LeaseExpired)
    async def expired(request, exc):
        return JSONResponse(status_code=503, content={"error": {"type": "lease_expired", "message": str(exc)}})

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.exception("Inference request failed")
        return JSONResponse(status_code=500, content={"error": {"type": "server_error",
                            "message": "Inference failed; inspect the tmux service log."}})

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    protected = [Depends(authenticate)]

    @app.get("/readyz", dependencies=protected)
    def ready():
        engine.check_lease()
        status = engine.status()
        return JSONResponse(status_code=200 if status["phase"] == "ready" else 503, content=status)

    @app.get("/v1/models", dependencies=protected)
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0,
                 "owned_by": "wurendi", "stage": spec["stage"],
                 "loaded": name == engine.model_id, "note": spec.get("note")}
                for name, spec in engine.specs.items()]}

    @app.post("/admin/load", dependencies=protected)
    def load(req: LoadRequest):
        return engine.load(req.model)

    @app.post("/admin/unload", dependencies=protected)
    def unload():
        if not engine.lock.acquire(blocking=False):
            raise BusyError("model is busy")
        try:
            engine._unload()
            engine.record("unloaded_by_api")
            return engine.status()
        finally:
            engine.lock.release()

    @app.post("/generate", dependencies=protected)
    def generate(req: GenerateRequest):
        prompt = req.prompt if req.prompt is not None else instruction_prompt(req.instruction, req.input)
        return engine.generate(prompt, model=req.model, max_new_tokens=req.max_new_tokens,
                               temperature=req.temperature, top_p=req.top_p, seed=req.seed, stop=req.stop)

    def completion(req, prompt, chat=False):
        stops = [req.stop] if isinstance(req.stop, str) else req.stop or []
        result = engine.generate(prompt, model=req.model, max_new_tokens=req.max_tokens,
                                 temperature=req.temperature, top_p=req.top_p, seed=req.seed, stop=stops)
        choice = {"index": 0, "finish_reason": result["finish_reason"]}
        if chat:
            choice["message"] = {"role": "assistant", "content": result["text"]}
        else:
            choice.update(text=result["text"], logprobs=None)
        return {"id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
                "object": "chat.completion" if chat else "text_completion", "created": int(time.time()),
                "model": result["model"], "choices": [choice], "usage": result["usage"]}

    @app.post("/v1/completions", dependencies=protected)
    def completions(req: CompletionRequest):
        return completion(req, req.prompt)

    @app.post("/v1/chat/completions", dependencies=protected)
    def chats(req: ChatRequest):
        return completion(req, chat_prompt([m.model_dump() for m in req.messages]), chat=True)

    return app
