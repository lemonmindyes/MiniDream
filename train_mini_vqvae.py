import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from mini_vqvae import Config, VQVAE


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # base train param
    batch_size = 256
    lr = 1e-3
    epoch = 100

    config = Config()
    model = VQVAE(config).to(device)
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total_param:{total_param}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    train_dataset = CIFAR10(root = 'data', train = True, transform = transform, download = True)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    opt = torch.optim.Adam(model.parameters(), lr = lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                              T_max = epoch * len(train_loader),
                                                              eta_min = lr / 100)

    for e in range(epoch):
        for step, (x, _) in enumerate(train_loader):
            x = x.to(device)
            embedding_loss, x_hat, perplexity = model(x)
            loss = torch.mean((x_hat - x) ** 2) + embedding_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            if step % 20 == 0:
                print(f'epoch:{e + 1}/{epoch} step:{step + 1}/{len(train_loader)} loss:{loss.item():.4f} '
                      f'lr:{lr_scheduler.get_last_lr()[0]:.6f}, perplexity:{perplexity.item():.4f}')
    torch.save(model.state_dict(), 'vqvae.pt')