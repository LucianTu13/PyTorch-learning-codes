import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class AntsBeesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.str_2_label = {"ants": 0, "bees": 1}
        self.img_info = []  # (path, label),...
        self.get_img_info()

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is an empty dir! Please check your path!".format(
                self.root_dir))
        return len(self.img_info)

    def get_img_info(self):
        class_names = ["ants", "bees"]

        for name in class_names:
            img_folder_name = name + "_image"
            # 拼接路径：.../train/ants_image
            img_list_path = os.path.join(self.root_dir, img_folder_name)
            img_label = self.str_2_label[name]

            # 增加一个容错判断，防止文件夹不存在报错
            if not os.path.isdir(img_list_path):
                print(f"Warning: Folder not found: {img_list_path}")
                continue

            img_list = os.listdir(img_list_path)
            for img_name in img_list:
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(img_list_path, img_name)
                    self.img_info.append((img_path, img_label))

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img)

        # 增加一步：确保图片是RGB格式（防止有PNG带有透明通道导致报错）
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label



if __name__ == "__main__":
    # root_dir 指向包含 "ants_image" 和 "bees_image" 的那个父文件夹
    root_dir = r"C:\Users\26491\Desktop\练手数据集\train"

    # 定义预处理
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 实例化数据集
    train_dataset = AntsBeesDataset(root_dir, transforms_train)
    print(f"数据集加载成功，共找到 {len(train_dataset)} 张图片。")

    # 实例化DataLoader
    train_loader_bs2 = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    train_loader_bs3 = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True)
    train_loader_bs2_drop = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, drop_last=True)
    # 简单的测试循环
    for i, (img, label) in enumerate(train_loader_bs2):
        print(f"Batch {i}: Img Shape {img.shape}, Label: {label}")
        break  # 只打印第一个batch看看，避免刷屏

    for i, (img, label) in enumerate(train_loader_bs3):
        print(f"Batch {i}: Img Shape {img.shape}, Label: {label}")
        break

    for i, (img, label) in enumerate(train_loader_bs2_drop):
        print(f"Batch {i}: Img Shape {img.shape}, Label: {label}")
        break