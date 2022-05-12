import numpy as np
import torch.nn as nn


# 生成器
class DCGenerator(nn.Module):
    def __init__(self, in_features, img_shape, init_weights=True):
        super().__init__()
        self.img_shape = img_shape

        # 因为默认在CIFAR10上训练，且默认第一个卷积层输入是3*4*4
        # 在每次上采样放大两倍宽高的情况下，只需要做3次上采样便得到了所需的图片大小
        # 所以这里我比原论文少了一层上采样，原论文目标图片宽高是64的
        conv_channels = [1024, 512, 256, 3]

        # 默认每次放大2倍宽高，用于上采样
        def upsampling_block(in_channel, out_channel, normalize=True, activation=None, kernel_size=4, stride=2, padding=1):
            layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True) if activation is None else activation)
            return layers

        self.linear = nn.Sequential(
            # BN层前面的层bias可以为False
            nn.Linear(in_features, conv_channels[0] * np.prod(self.img_shape[1:]), bias=False),
            nn.BatchNorm1d(conv_channels[0] * np.prod(self.img_shape[1:])),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            *upsampling_block(conv_channels[0], conv_channels[1]),      # 8 * 8
            *upsampling_block(conv_channels[1], conv_channels[2]),      # 16 * 16
            *upsampling_block(conv_channels[2], conv_channels[3], False, nn.Tanh())     # 32 * 32
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], *self.img_shape)     # 变换成二维用于卷积
        return self.net(x)


# 判别器
class DCDiscriminator(nn.Module):
    def __init__(self, img_shape, init_weights=True):
        super().__init__()

        conv_channels = [3, 256, 512, 1024, 1]

        # 默认每次缩小2倍宽高，用于下采样
        def downsampling_block(in_channel, out_channel, normalize=True, activation=None, padding=1):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        self.net = nn.Sequential(
            *downsampling_block(conv_channels[0], conv_channels[1], False),     # 16 * 16
            *downsampling_block(conv_channels[1], conv_channels[2]),            # 8 * 8
            *downsampling_block(conv_channels[2], conv_channels[3]),            # 4 * 4
            *downsampling_block(conv_channels[3], conv_channels[4], activation=nn.Sigmoid(), padding=0),
            #nn.AdaptiveAvgPool2d((1, 1)), nn.Sigmoid()
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images):
        y = self.net(images)
        return y.view(y.shape[0], -1)
