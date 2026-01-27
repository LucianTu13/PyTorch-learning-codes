import torch
from jinja2.nodes import Output

print(torch.linspace(3, 10,5))
print(torch.linspace(1, 5, 3))

# Output
# tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])
# tensor([1., 3., 5.])