import torch
import torch.nn as nn

# 1. 设置随机种子，保证每次运行结果一致
torch.manual_seed(42)

# 2. 定义超参数
seq_len = 50        # 序列长度（模拟长句子）
input_size = 10     # 输入特征维度

hidden_size = 20    # 隐藏层维度
# 3. 创建输入数据
# 注意：requires_grad=True 是关键，这样我们才能查看输入数据的梯度
inputs = torch.randn(1, seq_len, input_size, requires_grad=True)

print(f"实验设置：序列长度 = {seq_len}, 隐藏层大小 = {hidden_size}")
print("-" * 50)

# ==========================================
# 实验 A: 普通 RNN
# ==========================================
rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# 前向传播
output_rnn, _ = rnn(inputs)

# 定义 Loss：我们只关心最后一个时间步的输出
# 这模拟了“根据整个句子预测下一个词”或“情感分类”的任务
loss_rnn = output_rnn[:, -1, :].sum()

# 反向传播
loss_rnn.backward()

# 获取 t=0 (序列开头) 和 t=49 (序列结尾) 的梯度模长
grad_rnn_first = inputs.grad[:, 0, :].norm().item()
grad_rnn_last = inputs.grad[:, -1, :].norm().item()

print(f"【RNN 结果】")
print(f"  t=49 (结尾) 的梯度: {grad_rnn_last:.6f}")
print(f"  t=0  (开头) 的梯度: {grad_rnn_first:.6f}")
print(f"  衰减比例 (First/Last): {grad_rnn_first / grad_rnn_last:.6e}")

# ==========================================
# 实验 B: LSTM
# ==========================================
# 清空之前的梯度
inputs.grad.zero_()

lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# 前向传播
output_lstm, _ = lstm(inputs)

# Loss：同样只关心最后一个时间步
loss_lstm = output_lstm[:, -1, :].sum()

# 反向传播
loss_lstm.backward()

grad_lstm_first = inputs.grad[:, 0, :].norm().item()
grad_lstm_last = inputs.grad[:, -1, :].norm().item()

print("-" * 50)
print(f"【LSTM 结果】")
print(f"  t=49 (结尾) 的梯度: {grad_lstm_last:.6f}")
print(f"  t=0  (开头) 的梯度: {grad_lstm_first:.6f}")
print(f"  衰减比例 (First/Last): {grad_lstm_first / grad_lstm_last:.6e}")