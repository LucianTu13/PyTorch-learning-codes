import torch

# 1. 准备一个全 0 的目标张量 (3x5)
target = torch.zeros(3, 5)
print(target)

# 2. 准备我们要填进去的数据
src = torch.tensor([[10, 20, 30],
                    [40, 50, 60]],dtype=torch.float32) # 2x3

# 3. 准备索引：我们要把上面的数据填到 target 的哪些行/列？
# 假设我们沿着 dim=1 (横向) 分发
index = torch.tensor([[0, 2, 4],
                      [1, 3, 0]]) # 形状必须和 src 一致 (2x3)

# 4. 执行 scatter
# target[i][index[i][j]] = src[i][j]
target.scatter_(dim=1, index=index, src=src)

print(target)