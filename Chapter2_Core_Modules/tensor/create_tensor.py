import torch
import numpy as np

v = [[1.,-1.],[1.,-2.]]

a = torch.tensor(v)

print(a)
print(a.dtype)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")

b = np.array(v)
print(b)
print(b.dtype)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")

c = torch.tensor(b)

# numpy 使用float64存储数据，而PyTorch使用float32存储数据，转化时若是没有标注数据类型，则会在输出的时候强调torch.float64

print(c)
print(c.dtype)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# 改为float32适应PyTorch内部的数据类型可加快运行速率
d = torch.tensor(b,dtype=torch.float32)

print(d)
print(d.dtype)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
e = torch.from_numpy(b)

print(e)
print(e.dtype)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
