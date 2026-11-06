"""Common setup for adapter/router trainers, with no constructor side effects."""
from abc import ABC, abstractmethod
from models.backbones.qwen3 import Qwen3Backbone
from models.trainers.checkpoint import atomic_json, atomic_save


class BasicTrainer(ABC):
    """Own model loading, base freezing, adapter setup and optimizer creation.

    Stage subclasses keep their original losses, sampling, schedules and resume
    rules. Constructing a trainer never loads a model or touches a GPU.
    """

    def __init__(self, args, *, backbone=None):
        self.args = args
        self.backbone = backbone or Qwen3Backbone()
        self.model = None
        self.optimizer = None
        self.trainable_parameters = []

    def seed(self):
        import torch
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    @staticmethod
    def require_gpus():
        import torch
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No GPU visible inside container")
        return count

    def load_model(self, gpu_count, *, reject_offload=True):
        import torch
        model = self.backbone.load_pretrained(
            self.args.model, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={i: self.args.max_memory for i in range(gpu_count)},
            low_cpu_mem_usage=True, attn_implementation="eager")
        if reject_offload and any(d in ("cpu", "disk") for d in model.hf_device_map.values()):
            raise RuntimeError("Some base layers were offloaded; increase --max-memory")
        model.config.use_cache = False
        self.model = model
        return model

    def prepare_optimizer(self, model):
        import torch
        self.backbone.prepare_adapters(model, self.args.rank)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.args.lr, weight_decay=0.01)
        self.trainable_parameters, self.optimizer = parameters, optimizer
        return parameters, optimizer

    atomic_save = staticmethod(atomic_save)
    atomic_json = staticmethod(atomic_json)

    @abstractmethod
    def run(self):
        """Execute this stage only after an explicitly authorized launch."""
