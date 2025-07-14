from dataclasses import dataclass


@dataclass
class Config:
    # img param
    img_size: int = 224
    channel: int = 3
    patch_size: int = 16
    dropout_rate: float = 0.0
    num_class: int = 10
    # transformer param
    dim: int = 512
    n_head: int = 4
    n_layer: int = 4