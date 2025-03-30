<!-- markdownlint-disable MD024 -->
# Infinite Shakespeare

![language](https://img.shields.io/badge/language-Python-blue)
![pytorch](https://img.shields.io/badge/Framework-Pytorch-orange)

无限莎士比亚是一个类似`GPT`的简易`LLM`，项目的作者是`Andrej Karpathy`大佬。本仓库是我尝试复刻该项目的记录。我刚接触这个项目时是看了原作者的教学视频和示例代码，不过迟钝的我并没能完全理解项目的每一个细节，因此我希望能通过这个仓库自己再敲一遍代码，顺便在`README`中记录下自己对代码的理解。

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

>[!Tip]
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

### 多头注意力机制

多头注意力机制（`MultiHeadAttention`）是对单头注意力机制的扩展。它的核心思想是**通过多个注意力头（head）并行计算注意力，捕获输入序列中不同子空间的特征**。每个注意力头独立计算，然后将它们的结果拼接起来，形成更丰富的表示。

```python
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
```

简单来说，**多头注意力机制让模型能够从多个角度关注输入序列中的信息**。

#### 代码的主要步骤

##### 1. 初始化

- **`num_heads`**: 注意力头的数量。
- **`head_size`**: 每个注意力头的特征维度。
- **`self.heads`**: 包含多个单头注意力机制（`Head`）的列表，每个注意力头独立计算。
- **`self.proj`**: 一个线性层，用于将多个注意力头的输出拼接后映射回原始的特征维度。
- **`self.dropout`**: 用于防止过拟合的 Dropout 层。

##### 2. 前向传播

- 输入张量 `x` 的形状为 `(B, T, C)`：
  - `B`: 批次大小（batch size）。
  - `T`: 序列长度。
  - `C`: 每个位置的特征维度（ embedding 的维度）。

- **计算多个注意力头的输出**：
  - 遍历每个注意力头（`Head`），对输入 `x` 进行独立的注意力计算。
  - 每个注意力头的输出形状为 `(B, T, head_size)`。
  - 将所有注意力头的输出拼接在最后一个维度上，得到形状为 `(B, T, C * num_heads)` 的张量。

- **线性变换和 Dropout**：
  - 使用 `self.proj` 将拼接后的张量映射回原始的特征维度 `C`。
  - 应用 Dropout，防止过拟合。

- **返回结果**：
  - 输出张量的形状为 `(B, T, C)`，与输入形状一致，但每个位置的表示已经结合了多个注意力头的信息。

---

#### 栗子接上回

还是那句话：`"I love AI"`，输入张量 `x` 的形状为 `(1, 3, 4)`，表示 1 个句子（批次大小为 1），句子长度为 3，每个单词用 4 维向量表示。

##### 初始化

- 假设 `num_heads = 2`，`head_size = 4`。
- 每个注意力头会独立处理输入张量，并生成 4 维的输出。

##### 前向传播

1. **计算每个注意力头的输出**：
   - 第 1 个注意力头的输出形状为 `(1, 3, 4)`。
   - 第 2 个注意力头的输出形状为 `(1, 3, 4)`。

2. **拼接注意力头的输出**：
   - 将两个注意力头的输出拼接在最后一个维度上，得到形状为 `(1, 3, 8)` 的张量。

3. **线性变换和 Dropout**：
   - 使用 `self.proj` 将拼接后的张量映射回原始的特征维度（`4`）。
   - 应用 Dropout，得到最终的输出。

##### 输出

- 输出张量的形状为 `(1, 3, 4)`，与输入形状一致，但每个单词的表示已经结合了多个注意力头的信息。

##### 总结

多头注意力机制的核心是：

1. **并行计算多个注意力头**：每个注意力头独立计算，捕获不同的特征。
2. **拼接注意力头的输出**：将多个注意力头的结果组合起来，形成更丰富的表示。
3. **线性变换和 Dropout**：将拼接后的结果映射回原始特征维度，并应用 Dropout 防止过拟合。

它的作用是让模型能够从多个角度关注输入序列中的信息，从而更好地理解复杂的上下文关系。

### 前馈神经网络

接下来，定义`FeedFoward`前馈神经网络，它的结构相对简单，经过线性层 -> 激活函数 -> 线性层 -> Dropout 之后，如果同样以`"I love AI"`为例，输入（1，3，4）的张量在经过网络输出之后仍然为（1，3，4）的张量。

### Transformer基本块

之后我们利用之前定义的种种网络结构组合成一个`Transformer`基本块，更多有关`Transformer`的详细信息可以看看谷歌团队在2017年发表的[《 Attention Is All You Need 》](https://arxiv.org/abs/1706.03762)这篇论文。另外，在这一步中，除了用到之前定义的多注意力头和前馈神经网络之外，还利用残差连接和归一化确保训练的稳定性。

### BigramLanguageModel

终于到了激动人心的模型部分，这个模型其实主要是由我们上面定义出来的各种组件拼装而成的，这里只对与前文不同的地方做出解释。它有着这样的结构：

```python
def __init__(self):
   super().__init__()
   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
   self.position_embedding_table = nn.Embedding(block_size, n_embd)
   self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
   self.ln_f = nn.LayerNorm(n_embd)
   self.lm_head = nn.Linear(n_embd, vocab_size)
```

#### 嵌入层

定义了一个 **嵌入层（Embedding Layer）**，它的作用是将离散的字符索引（如 "I love AI" 中的字符索引）映射到一个连续的向量空间中。为后续的模型提供输入。

- token_embedding  
  将输入的字符索引（例如 `"I love AI"` 经过编码之后得到的内容）转化为向量表示，每一个字符会被用一个`n_embd`维的向量表示。比如在我们的字符级模型中`"I love AI"`编码之后的形状为（1，9），`token_embedding_table`是一个大小为`(vocab_size, n_embd)`的嵌入矩阵，它会根据索引查找对应的嵌入向量，让每一个字符被映射为一个`n_embd`维的向量，从而得到形状（1，9，n_embd）的向量。让模型能够感知字符的语义信息。
- position_embedding  
  位置嵌入层，作用是为输入序列中的每个位置（如第 1 个字符、第 2 个字符等）生成一个固定的向量表示。这是因为 Transformer 模型本身对序列的顺序没有内置的感知能力，需要通过位置嵌入来显式地告诉模型每个字符在序列中的位置。`position_embedding_table`是一个大小为`(block_size, n_embd)`的矩阵，即为序列长度`block_size`中的每一个位置生成一个唯一的`n_embd`维的向量，让模型能够感知字符的顺序信息。

#### Transformer模块序列

由多`Transformer`基本块组成的序列，通过堆叠多个模块可以让模型逐层提取到更高层次的特征，每层的输出再作为下一层的输入，逐步增强模型对序列的理解能力。

#### 前向传播

在 `BigramLanguageModel` 的 `forward` 方法中，定义了模型的前向传播过程。它的作用是将输入的字符索引序列（如 "I love AI"）通过嵌入、位置编码、Transformer 块等模块，最终输出每个位置上所有可能字符的概率分布（logits），并计算损失（如果提供了目标 `targets`）。

假设`n_embd = 64 ; vocab_size = 65`，输入是 "I love AI"，对应的索引序列为 `[10, 20, 30, 40, 50, 60, 70, 80, 90]`，形状为 `(1, 9)`（1 表示 batch size，9 表示序列长度）。

1. 输入 "I love AI"，假设索引序列为 `[10, 20, 30, 40, 50, 60, 70, 80, 90]`，形状为 `(1, 9)`。
2. 通过字符嵌入层，得到形状为 `(1, 9, 64)` 的嵌入向量。
3. 通过位置嵌入层，得到形状为 `(9, 64)` 的位置向量，并与字符嵌入相加。
4. 通过堆叠的 Transformer 块，提取上下文信息，输出形状为 `(1, 9, 64)`。
5. 通过层归一化和线性层，映射到词汇表大小，输出形状为 `(1, 9, 65)`。
6. 如果提供目标值，计算交叉熵损失。

其中每一个阶段的细节都与前面的内容相关，就不再赘述了。

#### 内容生成

最后是使用`generate()`来生成内容，大致思路就是取内容的最后一段与序列长度等长的内容输入网络并将预测输出的内容衔接在最后，循环往复，直到达到最大生成数量。另外我在原项目基础上增加了内容输出的功能，这样可以让模型的生成结果看起来是流式输出的。

### 模型训练

1. **评估模型性能**：
   - 每隔 100 轮或最后一轮，调用 `estimate_loss` 方法，计算模型在训练集和测试集上的损失（`loss`）。
   - 打印训练损失和测试损失，方便观察模型的训练进展。

2. **获取训练数据**：
   - 调用 `get_batch('train')` 方法，从训练集中随机采样一个批次的数据。
   - 返回两个张量：
     - `X`：输入序列，形状为 `(batch_size, block_size)`。
     - `Y`：目标序列，形状为 `(batch_size, block_size)`。

3. **前向传播**：
   - 将 `X` 和 `Y` 输入模型，得到预测值（`logits`）和损失值（`loss`）。
   - `logits` 是模型对每个位置上所有可能字符的预测分布。
   - `loss` 是预测值和目标值之间的误差。

4. **反向传播**：
   - 调用 `loss.backward()` 计算损失对模型参数的梯度。
   - 梯度表示每个参数需要调整的方向和幅度，以减少损失。

5. **更新参数**：
   - 调用 `optimizer.step()` 根据梯度更新模型的参数。
   - 调用 `optimizer.zero_grad(set_to_none=True)` 清空梯度，为下一次迭代做准备。

---

## 参考资料

- [Andrej Karpathy's Google Colab Link](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
