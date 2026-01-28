from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
from torchvision import transforms

class MyImageDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)

        # 定义一个转换器
        #  PIL图片 → Tensor → 归一化到[0,1]之间
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        file_name = self.img_list[index]
        file_path = os.path.join(self.img_dir,file_name)

        # 读取图像文件
        # 用 PIL 打开图片。convert('RGB') 是为了防止有些图片是黑白的(1通道)或者透明的(4通道)
        img_pil = Image.open(file_path).convert("RGB")

        # 普通的图像文件不能进神经网络进行训练，因此需要转换为Tensor
        img_tensor = self.transform(img_pil)

        label = "0" if "cat" in file_name else "1"

        return img_tensor,label

# 实例化
my_dataset = MyImageDataset(img_dir = "./fake_images")
img_tensor,label = my_dataset[0]
# print(img_tensor)
# print(label)

train_loader = DataLoader(dataset = my_dataset,batch_size=4,shuffle = True)
for i,(batch_imgs,batch_labels) in enumerate(train_loader):
    print(f"Batch 图片形状: {batch_imgs.shape}")
    print(f"Batch 标签: {batch_labels}")
    break