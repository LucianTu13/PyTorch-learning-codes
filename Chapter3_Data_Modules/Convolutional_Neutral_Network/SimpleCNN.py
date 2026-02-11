import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义一个卷积层：
        # in_channels=1 (黑白图片), out_channels=16 (我们要找16种不同的特征)
        # kernel_size=3 (3x3的滤波器)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

        # 定义一个激活函数 (ReLU)，增加非线性
        self.relu = nn.ReLU()

    def forward(self, x):
        # 数据流向： 输入 -> 卷积 -> 激活 -> 输出
        x = self.conv1(x)
        x = self.relu(x)
        return x


# 1. 实例化模型
model = SimpleCNN()

# 2. 查看一开始的滤波器（完全是随机的噪声）
print("学习前的随机滤波器片段:", model.conv1.weight[0][0])

# 3. 假装放入数据
dummy_input = torch.randn(1, 1, 6, 6)  # batch=1, channel=1, 6x6
output = model(dummy_input)

print(f"输出的特征图维度: {output.shape}")
# 结果应该是 [1, 16, 4, 4] -> 因为我们用了16个滤波器，所以输出了16张特征图