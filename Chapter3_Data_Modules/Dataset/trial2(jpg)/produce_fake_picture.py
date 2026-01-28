import os
import torch
from torchvision.utils import save_image

# 1. 定义设置
data_dir = "./fake_images"
num_images = 20  # 我们生成 20 张图片练手
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"🎨 开始生成 {num_images} 张随机图片...")

# 2. 循环生成
for i in range(num_images):
    # 模拟生成一张 3通道 (RGB), 64x64 大小的随机噪声图
    # torch.randn 生成的是正态分布的随机数
    img_tensor = torch.randn(3, 64, 64)

    # 模拟分类：前10张是猫(0)，后10张是狗(1)
    if i < 10:
        filename = f"cat_{i}.jpg"
    else:
        filename = f"dog_{i}.jpg"

    file_path = os.path.join(data_dir, filename)

    # save_image 是 PyTorch 自带的神器，直接把 Tensor 存成图片文件
    save_image(img_tensor, file_path)

print(f"✅ 图片已保存在 {data_dir} 文件夹中！")