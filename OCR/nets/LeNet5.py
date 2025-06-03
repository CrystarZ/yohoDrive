import torch.nn as nn


class OCRNUM(nn.Module):  # 2cov -> 2pool -> FC
    def __init__(self):
        super(OCRNUM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.ReLU(),  #! 激活
            nn.MaxPool2d(kernel_size=2),
            # 16,14,14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32,7,7
        )
        # FC
        self.out = nn.Linear(32 * 7 * 7, 10)

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

