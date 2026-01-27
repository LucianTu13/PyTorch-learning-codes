import torch

# Step1. 定义叶子节点
w = torch.tensor([1.0],requires_grad=True)
x = torch.tensor([2.0],requires_grad=True)

# Step2. 按计算图计算
a = torch.add(w,x)
b = torch.add(w,1)
y = torch.mul(a,b)

# Step3. 反向传播
y.backward()

# Step4. 计算梯度
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)