import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
sentence = "hello lstm"
# 构建字符表：{'h': 0, 'e': 1, 'l': 2, ...}
char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}
dic_char = {i: c for i, c in enumerate(char_set)}  # 用于解码

# 转换为数字序列
input_seq = [char_dic[c] for c in sentence[:-1]]  # 输入: hello lst
target_seq = [char_dic[c] for c in sentence[1:]]  # 目标: ello lstm

# 转换为 Tensor (Batch Size = 1)
# 维度格式: [Batch, Sequence Length]
inputs = torch.tensor([input_seq], dtype=torch.long)
targets = torch.tensor([target_seq], dtype=torch.long)


# 2. 定义 LSTM 模型
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = 128
        self.embedding_dim = 10

        # Embedding层：把数字变成向量
        self.embedding = nn.Embedding(len(char_set), self.embedding_dim)
        # LSTM层：input_size=10, hidden_size=128
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        # 全连接层：把记忆状态转化为预测概率
        self.fc = nn.Linear(self.hidden_size, len(char_set))

    def forward(self, x):
        # x shape: [1, 9] -> embeds shape: [1, 9, 10]
        embeds = self.embedding(x)
        # lstm_out shape: [1, 9, 128]
        lstm_out, _ = self.lstm(embeds)
        # output shape: [1, 9, 8] (8是字符表大小)
        output = self.fc(lstm_out)
        return output


model = SimpleLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 开始训练
print(f"目标句子: '{sentence}'\n")
print("开始训练...注意观察模型是如何'学会'拼写的：")
print("-" * 40)

for epoch in range(101):
    optimizer.zero_grad()
    outputs = model(inputs)

    # 调整维度以计算 Loss
    # outputs: [1, 9, 8] -> [9, 8], targets: [1, 9] -> [9]
    loss = criterion(outputs.view(-1, len(char_set)), targets.view(-1))
    loss.backward()
    optimizer.step()

    # 每 20 轮展示一次模型的预测结果
    if epoch % 20 == 0:
        # 取出预测概率最大的字符索引
        predicted_indices = torch.argmax(outputs, dim=2).squeeze().tolist()
        predicted_str = ''.join([dic_char[idx] for idx in predicted_indices])

        # 输入是 sentence[:-1] (h,e,l,l,o, ,l,s,t)
        # 我们希望输出是 sentence[1:] (e,l,l,o, ,l,s,t,m)
        # 把第一个字符 'h' 加上去，看看完整的句子长什么样
        full_prediction = sentence[0] + predicted_str

        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | 预测: {full_prediction}")

print("-" * 40)
print("训练完成！模型成功记住了字符的顺序和依赖关系。")