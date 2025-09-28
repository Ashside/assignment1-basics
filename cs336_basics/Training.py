import math
import os
import random
import typing
from collections.abc import Iterable
from typing import Optional, Callable

import torch
from numpy import typing as npt

import numpy as np

def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 数值稳定：先 log_softmax，再按标签索引负对数似然
    log_probs = logits.log_softmax(dim=-1)
    loss = -log_probs[torch.arange(logits.shape[0]),targets].mean()
    return loss

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr":lr}
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,params, lr=1e-3,weight_decay=0.01,betas=(0.9, 0.999),eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                t = t + 1

                grad = p.grad.data
                m = state.get("m",torch.zeros_like(grad))
                v = state.get("v",torch.zeros_like(grad))
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * (grad**2)
                at = lr * (math.sqrt(1 - (beta2 ** t))) / (1 - (beta1 ** t))
                p.data -= at * m / (torch.sqrt(v) + eps)
                p.data *= (1 - lr * weight_decay)
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss





def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters) * math.pi))*(max_learning_rate-min_learning_rate)/2
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total_norm += math.pow(param_norm,2)
    total_norm = math.sqrt(total_norm)
    coef = max_l2_norm / (total_norm + 1e-6)
    if coef < 1:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(coef)
    return parameters

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    full_lenth = len(dataset)
    indexs = [i for i in range(full_lenth - context_length) ]
    random.shuffle(indexs)
    pre_starts = indexs[:batch_size]
    pre_tokens =[]
    next_tokens=[]
    for i in range(batch_size):
        pre_tokens.append(dataset[pre_starts[i]:pre_starts[i]+context_length])
        next_tokens.append((dataset[pre_starts[i] + 1:pre_starts[i]+context_length +1]))
    pre_tokens_np = np.array(pre_tokens)
    next_tokens_np = np.array(next_tokens)
    return torch.from_numpy(pre_tokens_np).to(device), torch.from_numpy(next_tokens_np).to(device)

def save_checkpoint(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:str|os.PathLike |typing.BinaryIO |typing.IO[bytes]):
    model_dict = model.state_dict()
    optim_dict = optimizer.state_dict()
    d = {
        "model_dict":model_dict,
        "optim_dict":optim_dict,
        "iteration":iteration,
    }
    torch.save(d,out)

def load_checkpoint(
        src:str|os.PathLike |typing.BinaryIO |typing.IO[bytes],
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer
    ):
    d = torch.load(src)
    model_dict = d["model_dict"]
    optim_dict = d["optim_dict"]
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optim_dict)
    return d["iteration"]






