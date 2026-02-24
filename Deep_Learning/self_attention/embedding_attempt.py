from sentence_transformers import SentenceTransformer

# 这一行就是核心：加载模型并对文字进行编码
embedding = SentenceTransformer('all-MiniLM-L6-v2').encode("深度学习改变世界")

print(embedding) # 输出：一个包含 384 个数字的列表（向量）