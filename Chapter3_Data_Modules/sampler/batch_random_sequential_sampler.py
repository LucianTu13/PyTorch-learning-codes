from torch.utils.data import BatchSampler,RandomSampler,SequentialSampler

data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("--- 1. SequentialSampler (按顺序) ---")
# 仅仅是生成索引的迭代器
seq_sampler = SequentialSampler(data_source)
print(list(seq_sampler))

print("\n--- 2. RandomSampler (洗牌) ---")
rand_sampler = RandomSampler(data_source)
print(list(rand_sampler))

print("\n--- 3. BatchSampler (打包) ---")
# sampler 需要输入一个定义好的sampler
batch_sampler = BatchSampler(sampler = rand_sampler,batch_size=3,drop_last=False)
# 返回索引

for batch_indices in batch_sampler:
    print(batch_indices)