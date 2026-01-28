import os
from torch.utils.data import Dataset, DataLoader


class MyTxtDataset(Dataset):
    def __init__(self,root_dir):
        """
        初始化：这里只负责 "记录文件在哪里"，不负责读取具体内容
        """
        self.root_dir=root_dir

        # os.listdir 会扫描文件夹，返回一个包含所有文件名的列表 ['0.txt', '1.txt'...]
        self.file_list = os.listdir(root_dir)

        # 这一行比较高级，是为了解决文件名排序问题。
        # 电脑默认排序可能是 '1.txt', '10.txt', '2.txt'。
        # 这段代码强制按照数字大小排序：'0.txt', '1.txt', '2.txt'...
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.root_dir,file_name)

        with open(file_path,"r") as f:
            content = f.read()
            number = int(content)

            label = 0 if number%2 == 0 else 1

            return number,label

# 实例化
my_dataset = MyTxtDataset("./fake_data")
# 以下两种在PyTorch中是等价的做法
print(my_dataset.__getitem__(5))
print(my_dataset[5])

print(my_dataset.__len__())

# 再放入 Dataloader
train_loader = DataLoader(dataset = my_dataset,batch_size=4,shuffle = True)

for i,(data,label) in enumerate(train_loader):
    print(f"Batch {i}: 数据={data}, 标签={label}")

