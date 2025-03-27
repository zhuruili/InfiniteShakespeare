from math import log
import re
from turtle import pos
from sympy import real_root
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1024)

# 超参数设置
train_test_ratio = 0.8  # 训练集和测试集的比例
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8  # 批大小， 即（B, T, C）中的B
block_size = 32  # 序列长度， 即（B, T, C）中的T
learning_rate = 1e-3
eval_iters = 100  # 评估时的迭代次数
n_embd = 64  # embedding的维度， 即（B, T, C）中的C
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
        
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C * num_heads)
        out = self.dropout(self.proj(out))

        return out

class FeedFoward(nn.Module):
    """前馈神经网络"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer的一个基本模块"""

    def __init__(self, n_embd, n_head):
        """
        :param n_embd: embedding的维度
        :param n_head: 多头注意力机制的头数
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接和归一化
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))  

        return x
        
class BigramLanguageModel(nn.Module):
    """bigram语言模型"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        :param idx: 输入张量，形状为 (B, T)，B为batch_size，T为序列长度
        :param targets: 目标张量，形状为 (B, T)
        :return: logits, loss
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        """
        生成文本
        :param idx: 输入张量，形状为 (B, T)，B为batch_size，T为序列长度
        :param max_new_tokens: 生成的最大长度
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 只取最后一个block_size长度的序列
            logits, loss = self(idx_cond)  # 得到预测结果
            logits = logits[:,-1, :]  # 只取最后一个时间步的输出
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, 1)  # 从多项分布中采样
            idx = torch.cat([idx, idx_next], dim=-1)  # (B, T+1), 将预测结果拼接到序列后面

        return idx



