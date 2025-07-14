# MiniDream

> Implementing Sota CV, NLP, Speech and Multimodal Models with PyTorch

## 🌟 Features
- Pytorch
- Lightning

## 📦 Installation
```bash
git clone https://github.com/lemonmindyes/MiniDream.git
cd your_project
pip install -r requirements.txt
```

## 🚀 Usage
```bash
python train_mini_(model).py # model name
```

##  📚 Model
- [x] [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [x] [VIT](https://arxiv.org/abs/2010.11929)

### VQ-VAE
<img src = './imgs/VQVAE.jpg' width = '500px'></img>
```python
import torch
from mini_vqvae import Config, VQVAE

config = Config()
model = VQVAE(config)

img = torch.randn(1, 3, 224, 224)
embedding_loss, x_hat, perplexity = model(img) # x_hat [1, 3, 224, 224]
```

### VIT
<img src = './imgs/VIT.jpg' width = '500px'></img>
```python
import torch
from mini_vit import Config, VIT

config = Config()
config.num_class = 1000
model = VIT(config)

img = torch.randn(1, 3, 224, 224)
out = model(img) # out [1, num_class]
```




