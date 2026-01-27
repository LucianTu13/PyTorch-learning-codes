import torch

# 全局统一采样
x = torch.normal(mean=0.0, std=1.0, size=(2, 3))
print(x)

# 逐元素采样
means = torch.tensor([0.0, 10.0, 100.0])
stds = torch.tensor([1.0, 0.1, 0.01])

# 第1个元素来自 N(0, 1), 第2个来自 N(10, 0.1)...
y = torch.normal(means, stds)
print(y)

# 混合模式

z = torch.normal(mean=0.0,std=torch.arange(0.0,5.0))
print(z)

m = torch.normal(mean=torch.arange(0.0,5.0),std=1.0)
print(m)