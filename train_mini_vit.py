import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from mini_vit import Config, VIT


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # base train param
    batch_size = 256
    lr = 1e-3
    epoch = 30

    config = Config()
    config.img_size = 32
    config.patch_size = 4
    config.dim = 128
    config.dropout_rate = 0.0
    config.n_head = 4
    config.n_layer = 2
    model = VIT(config).to(device)
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total_param:{total_param}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CIFAR10(root='data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                              T_max=epoch * len(train_loader),
                                                              eta_min=lr / 100)

    for e in range(epoch):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = loss_func(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            if step % 50 == 0:
                print(f'epoch:{e + 1}/{epoch} step:{step + 1}/{len(train_loader)} loss:{loss.item():.4f} '
                      f'lr:{lr_scheduler.get_last_lr()[0]:.6f}')

        model.eval()
        total_acc = 0
        total_num = 0
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out = model(x)
            prob = torch.softmax(out, dim = -1)
            pred = torch.argmax(prob, dim = -1)
            total_acc += (pred == y).sum().item()
            total_num += x.shape[0]
        print(f'epoch:{e + 1}/{epoch} accuracy:{total_acc / total_num:.4f}')


    torch.save(model.state_dict(), 'vit.pt')
