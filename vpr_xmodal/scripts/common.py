"""Bits shared by the training scripts."""
import math
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import distributed as pml_dist


def add_common_args(ap, epochs: int, lr: float, lr_backbone: float, weight_decay: float, warmup: int):
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument("--lr_backbone", type=float, default=lr_backbone)
    ap.add_argument("--weight_decay", type=float, default=weight_decay)
    ap.add_argument("--warmup_steps", type=int, default=warmup)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--max_regions", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = full run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)


def make_accelerator(a) -> Accelerator:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # long timeout: the other processes wait while the main process runs validation
    acc = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no", gradient_accumulation_steps=a.grad_accum,
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))])
    set_seed(a.seed)
    return acc


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def ms_loss(acc: Accelerator, alpha: float, beta: float, base: float):
    loss = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base, distance=CosineSimilarity())
    return pml_dist.DistributedLossWrapper(loss) if acc.num_processes > 1 else loss


def distributed_miner(acc: Accelerator, miner):
    """Mines over the batch gathered from every process (pml's wrapper, used only when distributed)."""
    return pml_dist.DistributedMinerWrapper(miner) if acc.num_processes > 1 else miner


def ms_miner(acc: Accelerator, epsilon: float):
    return distributed_miner(acc, miners.MultiSimilarityMiner(epsilon=epsilon, distance=CosineSimilarity()))


class Meter:
    def __init__(self):
        self.sums, self.n = {}, 0

    def update(self, **kw):
        self.n += 1
        for k, v in kw.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)

    def reset(self):
        self.sums, self.n = {}, 0

    def __str__(self):
        return " ".join(f"{k}={v / max(self.n, 1):.4f}" for k, v in self.sums.items())


def save(acc: Accelerator, path, **state):
    if acc.is_main_process:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
