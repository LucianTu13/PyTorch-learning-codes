import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)

y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)    dy0/dw = 2w + x + 1
y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

loss = torch.cat([y0, y1], dim=0)       # [y0, y1]

grad_tensors = torch.tensor([1., 2.])

loss.backward(gradient=grad_tensors)    # Tensor.backward中的 gradient 传入 torch.autograd.backward()中的grad_tensors

# w =  1* (dy0/dw)  +   2*(dy1/dw)
# w =  1* (2w + x + 1)  +   2*(w)
# w =  1* (5)  +   2*(2)
# w =  9

print(w.grad)