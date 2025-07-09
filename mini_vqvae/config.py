from dataclasses import dataclass


@dataclass
class Config:
    # vae param
    in_dim: int = 3
    hidden_dim: int = 128
    res_dim: int = 32
    # vector quantizer param
    K: int = 512
    embedding_dim: int = 64
    beta: float = 0.25