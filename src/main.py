import torch

torch.manual_seed(1024)

# 超参数设置
train_test_ratio = 0.8  # 训练集和测试集的比例
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
block_size = 32

with open("data/tiny_Shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))  # 所有可能出现的字符
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符到索引的映射
itos = { i:ch for i,ch in enumerate(chars) }  # 索引到字符的映射
encode = lambda s: [stoi[c] for c in s]  # 字符串到索引序列的转换
decode = lambda x: ''.join([itos[i] for i in x])  # 索引序列到字符串的转换

# 训练集和测试集的划分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * train_test_ratio)
train_data = data[:n]
test_data = data[n:]

# 辅助函数
def get_batch(split):
    """
    从训练集或测试集中随机采样，生成`batch_size`个序列长度为`block_size`的样本
    :param split: 'train' or 'test'
    :return: x, y, 均为形状为 (batch_size, block_size) 的张量
    """
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



