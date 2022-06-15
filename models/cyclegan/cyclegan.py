import torch
import torch.nn as nn


def build_cbr_block(in_channels, out_channels, kernel_size, stride=1, padding=1,
                    activation=None, normalize=True, reflect_pad=True):
    layers = []

    if reflect_pad:
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False if normalize else True
        ),
    )

    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))

    if activation is not None:
        layers.append(activation)

    return layers


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, reflect_pad=True):
        super().__init__()

        self.block = nn.Sequential(
            *build_cbr_block(in_channels, in_channels, 3, 1, 1, nn.ReLU(True)),
            *build_cbr_block(in_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualGenerator(nn.Module):
    def __init__(self, in_channels=3, res_nums=6, reflect_pad=True, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 128, 64, 3]

        blocks = [
            *build_cbr_block(in_channels, conv_channels[0],
                             kernel_size=7, stride=1, padding=3, activation=nn.ReLU(True)),

            # downsampling
            *build_cbr_block(conv_channels[0], conv_channels[1],
                             kernel_size=3, stride=2, padding=1, activation=nn.ReLU(True), reflect_pad=False),
            *build_cbr_block(conv_channels[1], conv_channels[2],
                             kernel_size=3, stride=2, padding=1, activation=nn.ReLU(True), reflect_pad=False),
        ]

        for _ in range(res_nums):
            blocks.append(ResidualBlock(conv_channels[2], reflect_pad))

        # upsampling
        blocks += [
            nn.Upsample(scale_factor=2),
            *build_cbr_block(conv_channels[2], conv_channels[3],
                             kernel_size=3, stride=1, padding=1, activation=nn.ReLU(True), reflect_pad=False),
            nn.Upsample(scale_factor=2),
            *build_cbr_block(conv_channels[3], conv_channels[4],
                             kernel_size=3, stride=1, padding=1, activation=nn.ReLU(True), reflect_pad=False)
        ]

        # map to RGB
        blocks += [
            *build_cbr_block(conv_channels[4], conv_channels[5],
                             kernel_size=7, stride=1, padding=3, activation=nn.Tanh(), normalize=False)
        ]

        self.net = nn.Sequential(*blocks)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 512]

        def cbr_block(in_channel, out_channel, normalize=True, kernel_size=4, stride=2, padding=1, activation=None):
            layers = [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False if normalize else True),
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        # 感受野计算公式为 (output_size - 1) * stride + ksize
        # 倒着往上推就能算出感受野为70，最后一个output_size按1算
        self.net = nn.Sequential(
            *cbr_block(in_channels, conv_channels[0], normalize=False),
            *cbr_block(conv_channels[0], conv_channels[1]),
            *cbr_block(conv_channels[1], conv_channels[2]),
            *cbr_block(conv_channels[2], conv_channels[3], stride=1),
            *cbr_block(conv_channels[3], 1, normalize=False, stride=1, activation=nn.Sigmoid())
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
