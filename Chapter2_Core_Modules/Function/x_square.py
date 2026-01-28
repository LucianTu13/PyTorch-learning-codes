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
        ctx.save_for_backward(i) # save_for_backward()  存储张量

        return result

    @staticmethod
    def backward(ctx,grad_output):
        """
        :param ctx:
        :param grad_output:
        :return:
        """
        print("grad_output的值为：",grad_output)
        print("查看ctx.saved_tensors的值：",ctx.saved_tensors)
        i, = ctx.saved_tensors
        grad_i = 2 * i
        grad_results = grad_i * grad_output
        return grad_results

x = torch.tensor([3.],requires_grad=True)
a = x_square.apply(x)
print("前向传播结果是：",a)

a = a*10
a.backward()
print("对x的梯度为：",x.grad)
