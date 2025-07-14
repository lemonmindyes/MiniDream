import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from .config import Config
from .transformer import Transformer
from .utils import check_tuple


class VIT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.img_h, self.img_w = check_tuple(config.img_size)
        self.channel = config.channel
        self.ph, self.pw = check_tuple(config.patch_size)
        assert self.img_h % self.ph == 0 and self.img_w % self.pw == 0, f'img size must be divisible by patch size'
        self.patch_dim = self.channel * self.ph * self.pw
        self.dim = config.dim
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class

        # base module
        self.to_patch = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.ph, pw = self.pw),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.img_h // self.ph * self.img_w // self.pw, self.dim))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.transformer = Transformer(config)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim, self.num_class)
        )

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.to_patch(x)
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        cls = x[:, 0]
        out = self.to_out(cls)
        return out