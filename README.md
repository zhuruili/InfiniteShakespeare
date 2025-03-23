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

## 参考资料

- [Andrej Karpathy's Google Colab Link](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN)
