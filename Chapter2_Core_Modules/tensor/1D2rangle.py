import torch

src = torch.tensor([[1.],[2.],[3.],[4.]],dtype=torch.float32)

index = torch.tensor([[0],[1],[2],[3]])

target = torch.zeros((4,4))
print(target)

target.scatter_(dim=1,index=index,src=src)

print(target)

# Output:
# tensor([[1., 0., 0., 0.],
#         [0., 2., 0., 0.],
#         [0., 0., 3., 0.],
#         [0., 0., 0., 4.]])