from torch.utils.data import Dataset,DataLoader,ConcatDataset

class MyDataset(Dataset):
    def __init__(self,data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


data1 = MyDataset([1,2])
data2 = MyDataset([3,4,5])
data3 = MyDataset([6,7])

final_Data = ConcatDataset([data1,data2,data3])
print(len(final_Data))

loader = DataLoader(final_Data,batch_size=2,shuffle=True)
for batch in loader:
    print(batch)
# Output:
# 7
# tensor([2, 6])
# tensor([1, 3])
# tensor([4, 5])
# tensor([7])