import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from OCR.nets.LeNet5 import OCRNUM
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import cv2

cnn = OCRNUM()
# print(cnn)


def train():
    EPOCH = 100
    BATCH_SIZE = 50
    LR = 0.001  # 学习率
    DOWNLOAD_MNIST = False  # 自动下载

    # (28,28)->(1,28,28)
    transform = transforms.ToTensor()

    # data_set
    trainDataset = torchvision.datasets.MNIST(
        root="./data/", train=True, transform=transform, download=DOWNLOAD_MNIST
    )
    valDataset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=False
    )

    # 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
    train_loader = Data.DataLoader(
        dataset=trainDataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 打乱数据
    )

    # 进行测试
    # 为节约时间，测试时只测试前2000个
    val_loader = Data.DataLoader(dataset=valDataset, batch_size=2000, shuffle=False)
    test_x, test_y = next(iter(val_loader))

    # 训练
    # 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

    # 开始训练
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)  # 输出和真实标签的loss
            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float(
                    (pred_y == test_y.data.numpy()).astype(int).sum()
                ) / float(test_y.size(0))
                print(
                    "Epoch: ",
                    epoch,
                    "| train loss: %.4f" % loss.data.numpy(),
                    "| test accuracy: %.2f" % accuracy,
                )

        if epoch % 10 == 0:
            torch.save(cnn.state_dict(), "./.output/cnn2.pth")  # 保存模型


def test():
    TESTNUM = 32

    transform = transforms.ToTensor()
    testDataset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=False
    )
    test_loader = Data.DataLoader(
        dataset=testDataset, batch_size=TESTNUM, shuffle=False
    )
    test_x, test_y = next(iter(test_loader))

    cnn.load_state_dict(torch.load("./.output/cnn2.pth"))
    cnn.eval()
    # print 10 predictions from test data
    inputs = test_x  # 测试32个数据
    test_output = cnn(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()

    # 显示结果
    print(pred_y, "prediction number")  # 打印识别后的数字
    print(test_y.numpy(), "real number")

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    cv2.imshow("win", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()
