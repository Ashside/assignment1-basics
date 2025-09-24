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
        # 原地初始化，避免重绑定导致丢失 Parameter 身份
        mu = 0.0
        std = math.sqrt(2 / (in_features + out_features))
        a = -3 * std
        b = 3 * std
        nn.init.trunc_normal_(self.weight, mean=mu, std=std, a=a, b=b)


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

        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

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
    def __init__(self,
                 d_model,
                 d_ff,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        assert  2 <= d_ff / d_model <= 3 , r"d_ff is approximately  8/3 times d_model"
        assert d_ff % 64 == 0, "d_ff must be divisible by 64"
        self.d_model = d_model
        self.d_ff = d_ff


        self.w1 = Linear(d_ff,d_model,device=device,dtype=dtype)
         # w1: d_ff, d_model
        self.w2 = Linear(d_model,d_ff,device=device,dtype=dtype)
        # w2: d_model, d_ff
        self.w3 = Linear(d_ff,d_model,device=device,dtype=dtype)




    def forward(self,in_feature:torch.Tensor)->torch.Tensor:



        wx1 = self.w1(in_feature) # ... d_ff
        silu = wx1 * torch.sigmoid(wx1)
        wx3 = self.w3(in_feature) # ... d_ff
        x2 = silu * wx3 # ... d_ff
        wx2 = self.w2(x2) # ... d_model
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
        token_positions = token_positions.to(torch.int)
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
    # 需要将mask中为False的位置填充为-inf
    scores = scores.masked_fill(mask==False,float("-inf"))
    attn = softmax(scores,dim=-1)
    output = einsum(
        attn,V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 theta:float = None,
                 max_seq_len:int=None,
                 token_positions=None,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.wq = Linear(d_model,d_model,device=device,dtype=dtype)
        self.wk = Linear(d_model,d_model,device=device,dtype=dtype)
        self.wv = Linear(d_model,d_model,device=device,dtype=dtype)
        self.wo = Linear(d_model,d_model,device=device,dtype=dtype)


        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta,d_model//num_heads,max_seq_len,device=device)
            self.token_positions = token_positions


    def forward(self,Q,K,V):

        seq_len = Q.shape[1]
        xq,xk,xv = self.wq(Q),self.wk(K),self.wv(V) # (batch_size, seq_len, d_model)
        xq = rearrange(
            xq,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads = self.num_heads,
            d_k = self.d_k,
        )
        xk = rearrange(
            xk,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads = self.num_heads,
            d_k = self.d_k,
        )
        xv = rearrange(
            xv,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads = self.num_heads,
            d_k = self.d_k,
        )
        # 保留一个下三角为1，其余为0的掩码矩阵，默认包括主对角线
        attention_mask = torch.tril(torch.ones((seq_len,seq_len),dtype=torch.bool,device=Q.device))
        if hasattr(self,"rope"):
            token_positions = self.token_positions
            xq = self.rope(xq,token_positions)
            xk = self.rope(xk,token_positions)
        x = scaled_dot_product_attention(xq,xk,xv,attention_mask) # (batch_size, num_heads, seq_len, d_v)
        x = rearrange(
            x,
            "batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)"
        ) # (batch_size, seq_len, d_model)
        x = self.wo(x) # (batch_size, seq_len, d_model)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 max_seq_len:int,
                 theta:float,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len =max_seq_len
        self.theta = theta
        self.lm1 = RMSNorm(self.d_model,device=device,dtype=dtype)
        self.lm2 = RMSNorm(self.d_model,device=device,dtype=dtype)
        self.ffn = SwiGLU(self.d_model,self.d_ff,device=device,dtype=dtype)

        self.mha = MultiheadSelfAttention(
            self.d_model,
            self.num_heads,
            self.theta,
            self.max_seq_len,
            device=device,
            dtype=dtype
        )

        
    def forward(self, x:torch.Tensor):
        # x (batch_size, seq_len, d_model)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        assert self.d_model == x.shape[2]


        token_positons = torch.arange(seq_len,device=x.device,dtype=torch.int)
        self.mha.token_positions = token_positons


        y = x + self.mha(self.lm1(x),self.lm1(x),self.lm1(x))

        y_norm = self.lm2(y)

        output = y + self.ffn(y_norm)

        return output



class TransformerLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 context_length,
                 d_model,
                 num_layers,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device:torch.device|None=None,
                    dtype:torch.dtype|None=None
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.lm = RMSNorm(d_model,device=device,dtype=dtype)
        self.head = Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self, token_ids:torch.Tensor)->torch.Tensor:
        x = self.embedding(token_ids) # (batch_size, seq_len, d_model)
        for block in self.transformer_blocks:
            x = block(x) # (batch_size, seq_len, d_model)
        x = self.lm(x) # (batch_size, seq_len, d_model)
        logits = self.head(x) # (batch_size, seq_len, vocab_size)
        # 为什么这里不用softmax？？？？？？
        return logits






































