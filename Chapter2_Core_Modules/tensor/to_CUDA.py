import torch

v = [[1.,2.],[3.,4.]]
device = "cuda" if torch.cuda.is_available() else "cpu"

# 报错： 一旦tensor到了GPU，就不能和cpu上的tensor进行运算
a = torch.tensor(v).to(device)
b = torch.tensor(v)

print(a+b)
# "Expected all tensors to be on the same device"


