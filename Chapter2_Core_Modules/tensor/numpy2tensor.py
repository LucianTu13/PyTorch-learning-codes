import torch
import numpy as np

b = np.array([[1.,2.],[3.,4.]])
a1 = torch.from_numpy(b)
a2 = torch.tensor(b)

print(a1)
print(a2)

b[0,0] = 2

print(a1)
print(a2)
# tensor([[1., 2.],
#         [3., 4.]], dtype=torch.float64)
# tensor([[1., 2.],
#         [3., 4.]], dtype=torch.float64)
# tensor([[2., 2.],
#         [3., 4.]], dtype=torch.float64)
# tensor([[1., 2.],
#         [3., 4.]], dtype=torch.float64)