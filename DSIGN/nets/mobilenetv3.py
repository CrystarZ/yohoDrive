import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(  # FC+ReLU+FC+Sigmoid
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class conv_bn_hswish(nn.Module):
    def __init__(self, c1, c2, stride):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = h_swish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bneck(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:  # 通道扩张
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class MobileNetV3(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul):
        super().__init__()  # 3,640,640

        self.conv = conv_bn_hswish(3, 16, 2)  # 16,320,320
        self.dark2 = Bneck(16, 16, 16, 3, 2, 1, 0)  # 16,160,160
        self.dark3 = nn.Sequential(
            Bneck(16, 24, 72, 3, 2, 0, 0),
            Bneck(24, 24, 88, 3, 1, 0, 0),  # 24,80,80
        )
        self.dark4 = nn.Sequential(
            Bneck(24, 40, 96, 5, 2, 1, 1),
            Bneck(40, 40, 240, 5, 1, 1, 1),
            Bneck(40, 40, 240, 5, 1, 1, 1),
            Bneck(40, 48, 120, 5, 1, 1, 1),
            Bneck(48, 48, 144, 5, 1, 1, 1),  # 48,40,40
        )
        self.dark5 = nn.Sequential(
            Bneck(48, 96, 288, 5, 2, 1, 1),
            Bneck(96, 96, 576, 5, 1, 1, 1),
            Bneck(96, 96, 576, 5, 1, 1, 1),  # 96,20,20
        )

        self.c1 = nn.Conv2d(24, base_channels * 4, 1)  # 24 -> 256
        self.c2 = nn.Conv2d(48, base_channels * 8, 1)  # 48 -> 512
        self.c3 = nn.Conv2d(96, int(base_channels * 16 * deep_mul), 1)  # 96 -> 1024

    def forward(self, x):
        x = self.conv(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = self.c1(x)
        x = self.dark4(x)
        feat2 = self.c2(x)
        x = self.dark5(x)
        feat3 = self.c3(x)
        return feat1, feat2, feat3
