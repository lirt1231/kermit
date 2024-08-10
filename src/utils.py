from typing import Any, Dict, List, Tuple, Union

import torch
from torch.optim.lr_scheduler import LambdaLR


def report_model_parameters(model: torch.nn.Module) -> Tuple[float, float]:
    num_all = sum(p.nelement() for p in model.parameters())
    num_trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    return num_all, num_trainable


def move_to_cuda(data: Dict[Any, torch.Tensor], device: str = "cuda") -> Dict[Any, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True) if v is not None else None for k, v in data.items()
    }


def get_model_obj(model: Union[torch.nn.Module, torch.nn.DataParallel]) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_hits(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """Computes the Hit@k over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1 / batch_size).item())
    return res
