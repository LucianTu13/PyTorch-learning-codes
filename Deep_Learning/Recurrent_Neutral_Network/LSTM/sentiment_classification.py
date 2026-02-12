import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 准备数据 (微型数据集)
# ==========================================
# 1: 正面, 0: 负面
data = [
    ("i love this movie", 1),
    ("this is great", 1),
    ("fantastic model", 1),
    ("i hate this movie", 0),
    ("this is bad", 0),
    ("terrible result", 0)
]

# 构建词汇表
word_list = " ".join([text for text, _ in data]).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)


# 辅助函数：把句子变成数字 Tensor
def make_tensor(sentence):
    idxs = [word2idx[w] for w in sentence.split()]
    return torch.tensor([idxs], dtype=torch.long)


# ==========================================
# 2. 定义情感分析模型 (LSTM Classifier)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        # Embedding: 把单词索引变成稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM:
        # batch_first=True 意味着输入维度是 (batch, seq, feature)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Linear: 把 LSTM 的最终记忆映射为 1 个数值 (0~1之间)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 压缩到 0-1 变成概率

    def forward(self, x):
        # 1. 嵌入层
        # x: [batch, seq_len] -> embeds: [batch, seq_len, embed_dim]
        embeds = self.embedding(x)

        # 2. LSTM 层
        # output: 包含所有时间步的输出 (我们不需要)
        # (h_n, c_n): 包含最后一个时间步的隐藏状态 (我们需要这个！)
        _, (h_n, _) = self.lstm(embeds)

        # h_n 的维度是 [num_layers, batch, hidden_dim]
        # 我们取最后一层: [batch, hidden_dim]
        last_hidden_state = h_n[-1]

        # 3. 分类层
        y_pred = self.sigmoid(self.fc(last_hidden_state))
        return y_pred


# ==========================================
# 3. 训练模型
# ==========================================
model = LSTMClassifier(vocab_size, embed_dim=10, hidden_dim=16)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("开始训练情感分类器...")
for epoch in range(100):
    total_loss = 0
    for text, label in data:
        model.zero_grad()

        # 准备输入和目标
        input_tensor = make_tensor(text)
        target_tensor = torch.tensor([[float(label)]])  # 必须是 float

        # 前向传播
        pred = model(input_tensor)

        # 计算损失
        loss = criterion(pred, target_tensor)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# ==========================================
# 4. 测试模型
# ==========================================
print("\n--- 测试结果 ---")
test_sentences = ["love this", "hate this", "bad model"]

with torch.no_grad():
    for text in test_sentences:
        try:
            input_tensor = make_tensor(text)
            pred = model(input_tensor).item()
            sentiment = "正面 (Positive)" if pred > 0.5 else "负面 (Negative)"
            print(f"句子: '{text}' -> 评分: {pred:.4f} -> 判断: {sentiment}")
        except KeyError:
            print(f"句子: '{text}' 包含未知单词，无法测试。")