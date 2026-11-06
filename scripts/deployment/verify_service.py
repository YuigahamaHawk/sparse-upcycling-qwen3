"""HTTP functionality checks, optionally switching sequentially through stages.

This client is CPU-only; generation and model loading consume the service's
approved GPUs. Small smoke cases are not a capability benchmark.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:18082')
    p.add_argument('--api-key-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['moe-sft'])
    p.add_argument('--final-model', default='moe-sft')
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; choose a new result file')
    key = args.api_key_file.read_text().strip()
    results = {'purpose': 'service functionality only; not a model capability benchmark',
               'started_at_epoch': time.time(), 'models': [], 'checks': []}

    def call(path, payload=None, expected=200, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + key
        request = urllib.request.Request(args.url + path,
                  data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                code, data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            code, data = exc.code, json.loads(exc.read())
        if code != expected:
            raise RuntimeError(f'{path}: expected {expected}, got {code}: {data}')
        return data

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(args.output)

    try:
        until = time.monotonic() + 1200
        while True:
            try:
                call('/healthz', auth=False)
                break
            except urllib.error.URLError:
                if time.monotonic() >= until:
                    raise
                time.sleep(5)
        call('/v1/models', expected=401, auth=False)
        results['checks'].append('API key enforced')
        for model in args.models:
            print('VERIFY', model, flush=True)
            started = time.monotonic()
            ready = call('/admin/load', {'model': model})
            if ready['model'] != model or ready['phase'] != 'ready':
                raise RuntimeError('model did not become ready')
            raw = call('/v1/completions', {'model': model, 'prompt': 'The capital of France is', 'max_tokens': 12})
            instruction = call('/generate', {'model': model, 'instruction': '用一句话解释机器学习。', 'max_new_tokens': 32})
            if raw['model'] != model or instruction['model'] != model:
                raise RuntimeError('response model ID mismatch')
            if not 1 <= instruction['usage']['completion_tokens'] <= 32:
                raise RuntimeError('invalid generation token count')
            results['models'].append({'model': model, 'elapsed_seconds': time.monotonic()-started,
                                       'completion': raw, 'instruction': instruction})
            save()
        call('/admin/load', {'model': args.final_model})
        chat = call('/v1/chat/completions', {'model': args.final_model,
                    'messages': [{'role': 'user', 'content': 'Say hello briefly.'}], 'max_tokens': 16})
        results['chat'] = chat
        call('/v1/chat/completions', {'model': args.final_model,
             'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}, expected=422)
        call('/generate', {'prompt': 'x', 'max_new_tokens': 513}, expected=422)
        call('/generate', {'prompt': 'hello ' * 3000, 'max_new_tokens': 16}, expected=400)
        call('/admin/load', {'model': 'unregistered-model'}, expected=400)
        results['checks'].extend(['raw completion', 'instruction serialization', 'chat response',
                                  'streaming explicitly rejected', 'output cap enforced',
                                  'long input rejected without truncation', 'unregistered model rejected'])
        results['final_status'] = call('/readyz')
        results['passed'] = True
    except Exception as exc:
        results['passed'], results['error'] = False, str(exc)
        raise
    finally:
        results['finished_at_epoch'] = time.time()
        save()
    print(json.dumps({'passed': True, 'models': [x['model'] for x in results['models']],
                       'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
