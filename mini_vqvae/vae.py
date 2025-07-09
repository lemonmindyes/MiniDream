import torch.nn as nn

from .config import Config


class ResidualBlock(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.hidden_dim = config.hidden_dim
        self.res_dim = config.res_dim

        # base module
        self.res_block = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(self.hidden_dim, self.res_dim, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.res_dim, self.hidden_dim, 3, 1, 1)
        )

    def forward(self, x):
        # x: [b, c, h, w]
        out = x + self.res_block(x)
        return out


class Encoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim

        # base module
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_dim, self.hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            ResidualBlock(config),
            ResidualBlock(config),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [b, c, h, w]
        out = self.encoder(x)
        return out


class Decoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.in_dim = config.in_dim
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim

        # base module
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, self.hidden_dim, 3, 1, 1),
            ResidualBlock(config),
            ResidualBlock(config),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim // 2, self.in_dim, 4, 2, 1),
        )

    def forward(self, x):
        # x: [b, c, h, w]
        out = self.decoder(x)
        return out


