import torch
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)     # y = x**2

# 一阶导数
grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
print(grad_1)

# 二阶导数
grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
print(grad_2)