# 法一：使用numpy
import numpy as np

def cosine_similarity1(vec1,vec2):
    dot_product = np.dot(vec1,vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    cos_similarity = dot_product/(vec2_norm*vec1_norm)
    return cos_similarity

# 两个三维 embedding
a = np.array([1, 2, 3])
b = np.array([1, 9, 3.1])

print(f"这两个向量的余弦相似度为：{cosine_similarity1(a,b):.4f}")

# 法二：使用Scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
vec_a = [[1, 2, 3]]
vec_b = [[1, 2, 3.1]]

score = cosine_similarity(vec_a, vec_b)
print(score)
print(f"相似度: {score[0][0]}")