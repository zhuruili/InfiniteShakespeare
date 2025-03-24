# Infinite Shakespeare

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

## 参考资料

- [Andrej Karpathy's Google Colab Link](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN)
