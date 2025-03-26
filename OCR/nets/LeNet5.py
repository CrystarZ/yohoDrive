import torch.nn as nn

class OCRNUM(nn.Module): # 2cov -> 2pool -> FC
    def __init__(self):
        super(OCRNUM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 1,28,28
                in_channels=1,  # 图像通道数
                out_channels=16,  #! 输出通道数，卷积核个数
                kernel_size=5,  # 卷积核的大小5x5
                stride=1,  # 步长
                padding=2,  # padding = (kernel_size-1)/2
            ),  # 16,28,28
            nn.ReLU(),#! 激活
            nn.MaxPool2d(kernel_size=2),  #! 池化，下采样
            # 16,14,14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 16,14,14
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),# 32,14,14
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32,7,7
        )
        # FC
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类
        # self.out = nn.Linear(32 * 14 * 14, 10)  # 输出是10个类  输入56

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output