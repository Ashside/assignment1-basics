import math

import einops
from einops import rearrange,einsum
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features:int,
                 device:torch.device |None=None,
                 dtype:torch.dtype| None=None):
        """
        in_features: int final dimension of the input
        out_features:int final dimension of the output
        device: torch.device |None=None Device to store the parameters on
        dtype:torch.dtype| None=None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        # self.bias = nn.Parameter(torch.empty((out_features,),device=device,dtype=dtype))
        # 使用trunc_normal_初始化权重
        self.mu = 0.0
        self.std = math.sqrt(2/(in_features + out_features))
        self.a = -3*self.std
        self.b = 3*self.std
        with torch.no_grad():
            self.weight = nn.init.trunc_normal_(self.weight,mean=self.mu,std=self.std,a=self.a,b=self.b)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return einsum(
            x,self.weight,
            "... in_features, out_features in_features -> ... out_features"
        )





















































if __name__ == "__main__":
    channel_last = torch.randn(64,32,32,3)
    B =torch.randn(32*32,32*32)


    height = weight = 32
    channel_first = rearrange(
        channel_last,
        "b h w c -> b c (h w)"
    )
    print(channel_first.shape)
    channel_first_transformed = einsum(
        channel_first,B,
        "b c pixel_in,pixel_out pixel_in -> b c pixel_out"

    )
    print(channel_first_transformed.shape)
    channel_last_transformed = rearrange(
        channel_first_transformed,
        "b c (h w) -> b h w c",
        h=height,
        w=weight,
    )
    print(channel_last_transformed.shape)
