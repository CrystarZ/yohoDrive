import os
import sys
import matplotlib.pyplot as plt

pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(pwd)


def plot_loss_curve(file_path):
    # 读取损失值
    with open(file_path, "r") as f:
        lines = f.readlines()
        losses = [float(line.strip()) for line in lines if line.strip()]

    # 创建图像
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()


if __name__ == "__main__":
    PATH = "./.output/loss.txt"
    plot_loss_curve(PATH)
