import torch
from torch.autograd.function import Function

class GradCoeff(Function):

    @staticmethod
    def forward(ctx, x, coeff):

        # ============== step1: 函数功能实现 ==============
        ctx.coeff = coeff   # 将coeff存为ctx的成员变量
        x.view_as(x)
        # ============== step1: 函数功能实现 ==============
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print("grad_output的值为：",grad_output)
        return ctx.coeff * grad_output, None    # backward的输出个数，应与forward的输入个数相同，此处coeff不需要梯度，因此返回None

# 尝试使用
x = torch.tensor([2.], requires_grad=True)
print("x_value:",x)
ret = GradCoeff.apply(x, -0.1)                  # 前向需要同时提供x及coeff，设置coeff为-0.1
print("ret的值是：",ret)
ret = ret ** 3
print(ret)                                      # 注意看： ret.grad_fn
ret.backward()
print(x.grad)