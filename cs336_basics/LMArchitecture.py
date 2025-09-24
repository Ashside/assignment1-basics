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

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings:int, # size of the vocabulary
                 embedding_dim:int, # dimension of the embedding vectors ie d_model
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None
                 ):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mu = 0.0
        self.std = 1.0
        self.a = -3*self.std # 即[-3,3]之间
        self.b = 3*self.std
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))
        with torch.no_grad():
            self.weight = nn.init.trunc_normal_(self.weight,mean=self.mu,std=self.std,a=self.a,b=self.b)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = self.RMS(x)
        x_norm = x / rms
        result = x_norm * self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)

    def RMS(self,x:torch.Tensor)->torch.Tensor:
        assert x.shape[-1] == self.d_model
        return torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        assert  2 <= d_ff / d_model <= 3 , r"d_ff is approximately  8/3 times d_model"
        assert d_ff % 64 == 0, "d_ff must be divisible by 64"
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.ones(d_ff,d_model))
         # w1: d_ff, d_model
        self.w2 = nn.Parameter(torch.ones(d_model,d_ff))
        # w2: d_model, d_ff
        self.w3 = nn.Parameter(torch.ones(d_ff,d_ff))
        # w3: d_ff, d_ff



    def forward(self,in_feature:torch.Tensor)->torch.Tensor:

        wx1 = einsum(
            in_feature,self.w1,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        silu = wx1 * torch.sigmoid(wx1)
        wx3 = einsum(
            in_feature,self.w3,
            "... d_model, d_ff d_model-> ... d_ff"
        )
        x2 = silu * wx3 # ... d_ff
        wx2 = einsum(
            x2,self.w2,
            "... d_ff, d_model d_ff -> ... d_model"
        )
        return wx2

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device:torch.device |None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k

        self.max_seq_len = max_seq_len
        self.device = device

        i = rearrange(torch.arange(max_seq_len),"n -> n 1")
        k = rearrange(torch.arange(d_k//2),"d_kdiv2 -> 1 d_kdiv2")
        angle_rates = 1 / (theta ** (2 * k / d_k)) # (1, d_k/2)
        angle = i * angle_rates # (max_seq_len, d_k/2 )
        self.cosine = torch.cos(angle).to(device) # (max_seq_len, d_k/2 )
        self.sine = torch.sin(angle).to(device) # (max_seq_len, d_k/2 )
        self.register_buffer("cosine_buffer",self.cosine)
        self.register_buffer("sine_buffer",self.sine)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:

        cos = self.cosine_buffer[token_positions] # (batch_size, seq_len, d_k/2)
        sin = self.sine_buffer[token_positions] # (batch_size, seq_len, d_k/2)
        x_even = x[..., ::2] # (batch_size, seq_len, d_k/2)
        x_odd = x[..., 1::2] # (batch_size, seq_len, d_k/2)
        x_rotated_even = x_even * cos - x_odd * sin # (batch_size, seq_len, d_k/2)
        x_rotated_odd = x_even * sin + x_odd * cos # (batch_size, seq_len, d_k/2)
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1) # (batch_size, seq_len, d_k/2, 2)
        x_rotated = rearrange(x_rotated, "... d_kdiv2 two -> ... (d_kdiv2 two)")
        return x_rotated

def softmax(x:torch.Tensor,dim:int)->torch.Tensor:
    x_m = torch.max(x,dim=dim,keepdim=True).values
    x_exp = torch.exp(x - x_m)
    x_exp_sum = x_exp.sum(dim=dim,keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor):
    d_k = Q.shape[-1]
    scores = einsum(
        Q , K,
        "... queries d_k , ... keys d_k -> ... queries keys"
    )
    scores = scores / torch.sqrt(torch.tensor(d_k,dtype=scores.dtype))
    for i in range(len(mask.shape)-len(scores.shape)):
        mask = mask.unsqueeze(0)
    # 注意masked_fill_的逻辑和原mask矩阵不同
    scores = scores.masked_fill(mask==False,float("-inf"))
    attn = softmax(scores,dim=-1)
    output = einsum(
        attn,V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return output










































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
