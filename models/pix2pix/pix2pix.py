import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, down=True, activation=None, dropout=False):
        super().__init__()

        # 参数 4, 2, 1，在下采样是宽高缩小两倍，上采样时扩大两倍
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias=False if normalize else True) if down
            else nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False if normalize else True),
        )
        if normalize:
            self.net.append(nn.BatchNorm2d(out_channel))

        self.net.append(nn.LeakyReLU(0.2, True) if activation is None else activation)

        if dropout:
            self.net.append(nn.Dropout(0.5))

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 512, 512, 512, 512, 512, 512]

        self.down1 = UNetBlock(in_channels, conv_channels[0], down=True)
        self.down2 = UNetBlock(conv_channels[0], conv_channels[1], down=True)
        self.down3 = UNetBlock(conv_channels[1], conv_channels[2], down=True)
        self.down4 = UNetBlock(conv_channels[2], conv_channels[3], down=True)
        self.down5 = UNetBlock(conv_channels[3], conv_channels[4], down=True)
        self.down6 = UNetBlock(conv_channels[4], conv_channels[5], down=True)
        self.down7 = UNetBlock(conv_channels[5], conv_channels[6], down=True)

        self.bottleneck = UNetBlock(conv_channels[6], conv_channels[7], down=True)

        self.up1 = UNetBlock(conv_channels[7], conv_channels[6], down=False, activation=nn.ReLU(True))
        self.up2 = UNetBlock(conv_channels[6] * 2, conv_channels[5], down=False, activation=nn.ReLU(True), dropout=True)
        self.up3 = UNetBlock(conv_channels[5] * 2, conv_channels[4], down=False, activation=nn.ReLU(True))
        self.up4 = UNetBlock(conv_channels[4] * 2, conv_channels[3], down=False, activation=nn.ReLU(True), dropout=True)
        self.up5 = UNetBlock(conv_channels[3] * 2, conv_channels[2], down=False, activation=nn.ReLU(True))
        self.up6 = UNetBlock(conv_channels[2] * 2, conv_channels[1], down=False, activation=nn.ReLU(True), dropout=True)
        self.up7 = UNetBlock(conv_channels[1] * 2, conv_channels[0], down=False, activation=nn.ReLU(True))

        self.out = UNetBlock(conv_channels[0] * 2, in_channels, normalize=False, down=False, activation=nn.Tanh())

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        d1 = self.down1(x)      # 假设x.shape = (N, 3, 512, 512), d1.shape = （N, 64, 256, 256)
        d2 = self.down2(d1)     # (N, 128, 128, 128)
        d3 = self.down3(d2)     # (N, 256, 64, 64)
        d4 = self.down4(d3)     # (N, 512, 32, 32)
        d5 = self.down5(d4)     # (N, 512, 16, 16)
        d6 = self.down6(d5)     # (N, 512, 8, 8)
        d7 = self.down7(d6)     # (N, 512, 4, 4)

        bottleneck = self.bottleneck(d7)            # (N, 512, 2, 2)

        u1 = self.up1(bottleneck)                   # (N, 512, 4, 4)
        u2 = self.up2(torch.cat((u1, d7), 1))       # (N, 512, 8, 8)
        u3 = self.up3(torch.cat((u2, d6), 1))       # (N, 512, 16, 16)
        u4 = self.up4(torch.cat((u3, d5), 1))       # (N, 512, 32, 32)
        u5 = self.up5(torch.cat((u4, d4), 1))       # (N, 256, 64, 64)
        u6 = self.up6(torch.cat((u5, d3), 1))       # (N, 128, 128, 128)
        u7 = self.up7(torch.cat((u6, d2), 1))       # (N, 64, 256, 256)
        return self.out(torch.cat((u7, d1), 1))     # (N, 3, 512, 512)


# 默认 70x70 的感受野（patch）
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, init_weights=True):
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

    def forward(self, x, y):
        return self.net(torch.cat((x, y), 1))
