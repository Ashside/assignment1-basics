import math
from typing import Optional, Callable

import torch
from torch import nn
from cs336_basics.LMArchitecture import softmax

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












if __name__ == "__main__":
    weight = torch.nn.Parameter(5*torch.randn((10,10)))
    # print("weight:",weight)
    opt = SGD([weight],lr = 10)
    for t in range(100):
        opt.zero_grad() # 清空梯度
        loss = (weight ** 2).mean() # 计算损失
        loss.backward() # 反向传播计算梯度
        opt.step() # 更新参数
        if t % 10 == 0:
            print(f"step {t}, loss {loss.item():.4f}")