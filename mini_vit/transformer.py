import torch
import torch.nn as nn
from einops import rearrange

from .config import Config


class Attention(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.n_head = config.n_head
        self.dim = config.dim
        assert self.dim % self.n_head == 0, f'dim must be divisible by n_head'
        self.head_dim = self.dim // self.n_head
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = config.dropout_rate

        # base module
        self.to_qkv = nn.Linear(self.dim, 3 * self.n_head * self.head_dim, bias = False)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.to_out = nn.Linear(self.n_head * self.head_dim, self.dim)

    def forward(self, x):
        # x: [b, n, d]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_head), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.dim = config.dim
        self.dropout_rate = config.dropout_rate

        # base module
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):
        # x: [b, n, d]
        return self.ffn(x)


class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.dim = config.dim
        self.n_layer = config.n_layer

        # base module
        self.attn = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(self.dim),
            Attention(config)
        ) for _ in range(self.n_layer))
        self.ffn = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(self.dim),
            FeedForward(config)
        ) for _ in range(self.n_layer))

    def forward(self, x):
        # x: [b, n, d]
        for attn, ffn in zip(self.attn, self.ffn):
            x = attn(x) + x
            x = ffn(x) + x
        return x




