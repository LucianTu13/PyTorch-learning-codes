import os

# 1. 创建一个文件夹模拟 "硬盘上的数据集目录"
data_dir = "./fake_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 2. 生成 100 个虚拟文件
# 文件名是 0.txt, 1.txt ... 99.txt
# 文件内容就是一个随机数字
import random
for i in range(100):
    file_path = os.path.join(data_dir, f"{i}.txt")
    # 写入一个随机数 (0-1000)
    with open(file_path, "w") as f:
        f.write(str(random.randint(0, 1000)))

print("✅ 假数据生成完毕！文件夹位置:", data_dir)