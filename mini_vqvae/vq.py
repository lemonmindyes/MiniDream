import torch
import torch.nn as nn

from .config import Config


class VectorQuantizer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.k = config.K
        self.embedding_dim = config.embedding_dim
        self.beta = config.beta

        # base module
        self.embedding = nn.Embedding(self.k, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.k, 1.0 / self.k)

    def forward(self, x):
        # x: [b, c, h, w]
        z = x.permute(0, 2, 3, 1).contiguous()
        z_flatten = z.view(-1, self.embedding_dim)
        # (z - e)^2 = z^2 + e^2 - 2ze
        distance = (torch.sum(z_flatten ** 2, dim = 1, keepdim = True) +
                    torch.sum(self.embedding.weight ** 2, dim = 1) -
                    2 * torch.matmul(z_flatten, self.embedding.weight.t()))
        # find the nearest embedding
        min_idx = torch.argmin(distance, dim = 1).unsqueeze(dim = 1)
        min_encoding = torch.zeros(min_idx.shape[0], self.k).to(z.device)
        min_encoding.scatter_(1, min_idx, 1)
        # get latent vector
        z_q = torch.matmul(min_encoding, self.embedding.weight).view(z.shape)

        # loss
        embedding_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradient
        z_q = z + (z_q - z).detach()
        # perplexity
        e_mean = torch.mean(min_encoding, dim = 0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return embedding_loss, z_q, perplexity