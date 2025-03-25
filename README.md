# Infinite Shakespeare

![language](https://img.shields.io/badge/language-Python-blue)
![pytorch](https://img.shields.io/badge/Framework-Pytorch-orange)

无限莎士比亚是一个类似`GPT`的简易`LLM`，项目的作者是`Andrej Karpathy`大佬。本仓库是我尝试复该项目的记录。我刚接触这个项目时是看了原作者的教学视频和示例代码，不过迟钝的我并没能完全理解项目的每一个细节，因此我希望能通过这个仓库自己再敲一遍代码，顺便在`README`中记录下自己对代码的理解。

---

## 过程解读

光把代码跟着敲一遍感觉映像还是不够深刻，我希望能把对代码的理解记录在`README`中来巩固知识。

>[!Note]
>本章节的内容仅代表个人对于该项目源代码的粗浅理解，可能会有很多有误的地方，欢迎交流与赐教

### 输入输出

程序的输入是用来作为训练集的文本，在这里是`tiny_Shakespeare.txt`文件，里面的内容是形如人类对话的形式：

```txt
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
```

在经过训练后我们希望得到的输出是类似于这样对话形式的文本，但是其内容是由我们训练好的模型一个个输出的，形式和风格应该类似于`tiny_Shakespeare.txt`中的内容，但又不完全相同。

### 编码解码

在读取文件之后，我们通过`chars`查看所有可能出现的字符，`vocab_size`记录它们的个数。在本项目中我们要创建的是一个字符级的`LLM`，所以我们直接利用`lambda表达式`建立`encode`和`decode`来实现字符和编码之间的相互转化，例如：

```python
print(encode('InfiniteShakespeare'))
print(decode([21, 52, 44, 47, 52, 47, 58, 43, 31, 46, 39, 49, 43, 57, 54, 43, 39, 56, 43]))
```

第一行会输出字符串对应的编码序列，也就是第二行中的数字列表；第二行则反之。

### 划分采样

随后将整个数据集划分为训练集和测试集（我这里采取的比例是8:2），然后使用`get_batch()`函数进行随机采样，每次会生成`batch_size`个序列长度为`block_size`的样本，例如：

```python
batch_size = 2
block_size = 4
print(get_batch('train'))
```

结果是如下的一组`x`与`y`：

```python
(
    tensor(
        [[52, 58, 39, 45],[52, 42,  1, 47]], 
        device='cuda:0'
        ), 
    tensor(
        [[58, 39, 45, 59],[42,  1, 47, 57]], 
        device='cuda:0'
        )
)
```

### 损失计算

使用`estimate_loss()`函数评估模型在训练集和测试集上的平均损失。通过多次采样批次数据并计算损失，最终返回训练集和测试集的平均损失值，衡量模型的性能。

### 单头注意力机制

单头注意力机制的核心目标是**让模型能够关注输入序列中不同位置的信息，并根据这些信息生成上下文相关的输出**。它通过计算输入序列中每个位置之间的相关性（注意力权重），然后根据这些权重对输入进行加权求和，生成新的表示。

```python
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
```

简单来说，**它帮助模型理解序列中哪些部分对当前任务更重要**。

#### 代码的主要步骤

1. **输入张量**：
   输入是一个形状为 `(B, T, C)` 的张量：
   - `B`: 批次大小（batch size）。
   - `T`: 序列长度（例如一个句子的长度）。
   - `C`: 每个位置的特征维度（ embedding 的维度）。

2. **生成 `key`、`query` 和 `value`**：
   - `key`（键）：表示序列中每个位置的特征，用于被其他位置查询。
   - `query`（查询）：表示序列中每个位置想要查询的信息。
   - `value`（值）：表示序列中每个位置的实际内容。

   它们通过三个线性变换（`self.key`、`self.query`、`self.value`）从输入张量中生成，形状仍然是 `(B, T, head_size)`。

3. **计算注意力权重**：
   - 通过 `query` 和 `key` 的点积计算每个位置之间的相关性（注意力分数），结果是一个形状为 `(B, T, T)` 的矩阵，表示序列中每个位置对其他位置的注意力权重。
   - 使用 `masked_fill` 对未来的时间步进行掩码（mask），确保模型不会看到未来的信息（因果注意力）。
   - 对注意力分数进行 `softmax`，将其归一化为概率分布。

4. **加权求和**：
   - 使用注意力权重矩阵对 `value` 进行加权求和，生成新的表示，形状为 `(B, T, head_size)`。

5. **输出**：
   - 输出是一个与输入形状相同的张量 `(B, T, head_size)`，但每个位置的表示已经结合了序列中其他位置的信息。

#### 简单举个栗子

假设我们有一个句子：`"I love AI"`，我们希望模型能够理解每个单词之间的关系，因为我们不希望模型在学会了`I`、`love`、`AI`三个单词之后却说出形如`love AI I`这样的话

>[!info]
>在这个例子中出现的具体数值只是个示例，不一定代表真实的计算结果，这里是为了方便理解单头注意力机制到底在干什么

##### 输入

- 输入张量 `x` 的形状是 `(1, 3, 4)`，表示 1 个句子（批次大小为 1），句子长度为 3，每个单词用 4 维向量表示。

##### 生成 `key`、`query` 和 `value`

- 通过线性变换生成 `key`、`query` 和 `value`，它们的形状都是 `(1, 3, 4)`。

##### 计算注意力权重

- 计算 `query` 和 `key` 的点积，得到注意力分数矩阵：

  ```python
  wei = [[1.0, 0.8, 0.2],
         [0.8, 1.0, 0.5],
         [0.2, 0.5, 1.0]]
  ```

  这表示每个单词对其他单词的相关性。例如：
  - 第 1 个单词（`I`）对第 2 个单词（`love`）的相关性是 0.8。
  - 第 3 个单词（`AI`）对第 1 个单词（`I`）的相关性是 0.2。

- 使用 `softmax` 将分数归一化为概率：

  ```python
  wei = [[0.441, 0.361, 0.198],
         [0.338, 0.412, 0.250],
         [0.218, 0.295, 0.487]]
  ```

##### 加权求和

- 使用注意力权重对 `value` 进行加权求和，生成新的表示。例如：
  - 第 1 个单词的输出是：
  
    ```python
    out[0] = 0.441 * value[0] + 0.361 * value[1] + 0.198 * value[2]
    ```

##### 输出

- 输出张量的形状仍然是 `(1, 3, 4)`，但每个单词的表示已经结合了其他单词的信息。

##### 总结

单头注意力机制的核心是：

1. **计算相关性**：通过 `query` 和 `key` 的点积，找到序列中每个位置之间的关系。
2. **加权求和**：根据相关性对 `value` 进行加权求和，生成新的表示。

它的作用是让模型能够动态地关注序列中重要的信息，而不是简单地依赖固定的窗口或顺序。

---

## 参考资料

- [Andrej Karpathy's Google Colab Link](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
