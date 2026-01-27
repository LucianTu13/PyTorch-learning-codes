import torch
from torch.autograd.function import Function

class x_square(Function):
    @staticmethod
    def forward(ctx,i):
        """
        :param ctx:
        :param i:
        :return:
        """
        result = i**2
        ctx.save_for_backward(i)

        return result

    @staticmethod
    def backward(ctx,grad_output):
        """
        :param ctx:
        :param grad_output:
        :return:
        """
        print(grad_output)
        i, = ctx.saved_tensors
        grad_i = 2 * i
        grad_results = grad_i * grad_output
        return grad_results

x = torch.tensor([3.],requires_grad=True)
a = x_square.apply(x)
print(a)

a.backward()
print(x.grad)
