import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1024)

# 超参数设置
train_test_ratio = 0.8  # 训练集和测试集的比例
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
block_size = 32
eval_iters = 100  # 评估时的迭代次数
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

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

# 数据加载
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

# loss计算
@torch.no_grad()  # 使用`torch.no_grad`装饰器，避免梯度计算，节省内存和计算资源，因为评估时不需要反向传播
def estimate_loss(model):
    """
    估计模型在训练集和测试集上的loss
    :param model: 模型
    :return: 字典，包含训练集和测试集上的loss
    """
    out = {}
    model.eval()

    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)  # 用于存储每次迭代的loss
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)  # logits是模型的输出，loss是损失
            losses[i] = loss.item()
        out[split] = losses.mean()
    
    model.train()

    return out

class Head(nn.Module):
    """单头注意力机制"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (B, T, C)，B为batch_size，T为序列长度，C为特征维度（embedding的维度）
        :return: 输出张量，形状为 (B, T, C)
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C ** (-0.5)  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 掩码，避免未来信息泄露
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T, C)

        return out
        


