import torch

a = torch.rand((1,3))
print(a)


b = torch.rand_like(input=a)
print(b)

c = torch.randint(0,2,size=(2,3))
print(c)

d = torch.randn(size=(2,3))
print(d)

e = torch.bernoulli(input=torch.tensor([0.5,0.4,0.3]))
print(e)