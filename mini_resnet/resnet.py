import torch.nn as nn

from .config import Config


class BasicBlock(nn.Module):

    def __init__(self, in_dim, out_dim, stride):
        super().__init__()

        # base param
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride

        # base module
        self.conv1 = nn.Conv2d(self.in_dim, self.out_dim, kernel_size = 3, stride = self.stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(self.out_dim)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size = 1, stride = self.stride, padding = 0),
            nn.BatchNorm2d(self.out_dim)
        ) if self.stride != 1 or self.in_dim != self.out_dim else None

    def forward(self, x):
        # x: [b, c, h, w]
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleBlock(nn.Module):

    def __init__(self, in_dim, out_dim, stride):
        super().__init__()

        # base param
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = self.out_dim // 4
        self.stride = stride

        # base module
        self.conv1 = nn.Conv2d(self.in_dim, self.inter_dim, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(self.inter_dim)
        self.conv2 = nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size = 3, stride = self.stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.inter_dim)
        self.conv3 = nn.Conv2d(self.inter_dim, self.out_dim, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(self.out_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size = 1, stride = self.stride, padding = 0),
            nn.BatchNorm2d(self.out_dim)
        ) if self.stride != 1 or self.in_dim != self.out_dim else None
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        # x: [b, c, h, w]
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.channel = config.channel
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.resnet_name = config.resnet_name
        self.num_class = config.num_class
        if self.resnet_name == 'resnet18' or self.resnet_name == 'resnet34':
            self.res_expand = [1, 1, 2, 4]
        elif self.resnet_name == 'resnet50' or self.resnet_name == 'resnet101' or self.resnet_name == 'resnet152':
            self.res_expand = [1, 4, 8, 16]

        # base module
        self.conv1 = nn.Conv2d(self.channel, self.dim, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(self.dim, 64, 1, self.n_layer[0], self.res_expand[0])
        self.layer2 = self._make_layer(self.dim, 128, 2, self.n_layer[1], self.res_expand[1])
        self.layer3 = self._make_layer(self.dim, 256, 2, self.n_layer[2], self.res_expand[2])
        self.layer4 = self._make_layer(self.dim, 512, 2, self.n_layer[3], self.res_expand[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if self.resnet_name == 'resnet18' or self.resnet_name == 'resnet34':
            self.fc = nn.Linear(512, self.num_class)
        else:
            self.fc = nn.Linear(512 * 4, self.num_class)

    def _make_layer(self, in_dim, out_dim, stride, n_layer, expand):
        if self.resnet_name == 'resnet18' or self.resnet_name == 'resnet34':
            block = BasicBlock
            in_dim *= expand
        elif self.resnet_name == 'resnet50' or self.resnet_name == 'resnet101' or self.resnet_name == 'resnet152':
            block = BottleBlock
            in_dim *= expand
            out_dim *= 4
        else:
            raise ValueError('resnet name error')
        layers = [block(in_dim, out_dim, stride)]
        for _ in range(1, n_layer):
            layers.append(block(out_dim, out_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
