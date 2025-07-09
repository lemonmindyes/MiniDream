import torch.nn as nn

from .config import Config
from .vae import Encoder, Decoder
from .vq import VectorQuantizer


class VQVAE(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim

        # base module
        self.encoder = Encoder(config)
        self.conv1 = nn.Conv2d(self.hidden_dim, self.embedding_dim, 3, 1, 1)
        self.vq = VectorQuantizer(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        # x: [b, c, h, w]
        z = self.encoder(x)
        z = self.conv1(z)
        embedding_loss, z_q, perplexity = self.vq(z)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity