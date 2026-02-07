import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class RecoveryDataset(Dataset):
    def __init__(self, root_dir):
        # 1. 这里通常做什么？
        # (A) 读取所有图片的内容到内存
        # (B) 只记录所有图片的路径，不读内容
        self.img_paths = root_dir
        self.img_list = os.listdir(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # 2. 这里返回什么？
        # 答：返回文件夹的长度
        return len(self.img_list)

    def __getitem__(self, index):
        # 3. 这里是核心，三步走：
        # Step 1: 拿到第 index 个图片的路径
        # Step 2: 用 PIL 打开图片 (并可能做 transform)
        # Step 3: 返回什么？
        file_name = self.img_list[index]
        file_path = os.path.join(self.img_paths,file_name)
        img_pil = Image.open(file_path).convert("RGB")
        img_tensor = self.transform(img_pil)
        label = 0 if "cat" in file_name else 1
        return img_tensor,label

my_recovery_dataset = RecoveryDataset(root_dir = "./fake_images")

train_loader = DataLoader(dataset=my_recovery_dataset,batch_size=4,shuffle=True)
