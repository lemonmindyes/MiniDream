from dataclasses import dataclass, field


@dataclass
class Config:
    channel: int = 3
    dim: int = 64
    n_layer: list = field(default_factory = lambda: [2, 2, 2, 2])
    resnet_name: str = 'resnet18'
    num_class: int = 1000