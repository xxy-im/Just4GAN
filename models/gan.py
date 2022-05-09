import numpy as np
import torch.nn as nn


# 生成器
class Generator(nn.Module):
    def __init__(self, in_features, img_shape, init_weights=True):
        super().__init__()
        self.img_shape = img_shape

        def vanilla_block(in_feat, out_feat, normalize=True, activation=None):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU(inplace=True) if activation is None else activation)
            return layers

        self.net = nn.Sequential(
            *vanilla_block(in_features, 256, False),
            *vanilla_block(256, 512, True),
            *vanilla_block(512, 1024, True),
            *vanilla_block(1024, 1024, True),
            *vanilla_block(1024, np.prod(self.img_shape), False, nn.Tanh())
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, z):
        gz = self.net(z)
        return gz.view(-1, *self.img_shape)


# 判别器
class Discriminator(nn.Module):
    def __init__(self, img_shape, init_weights=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(img_shape), 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images):
        x = images.view(images.shape[0], -1)
        return self.net(x)

