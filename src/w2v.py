from gensim.models import Word2Vec
import re

# 读取文本并按行分句
with open("data\\tiny_Shakespeare.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 分词，每行为一个句子，并保留标点
sentences = [re.findall(r"\b\w+\b|[^\w\s]", line) for line in lines if line.strip()]

# 训练Word2Vec模型
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=32, # 词向量维度
    window=3,       # 上下文窗口大小
    min_count=1,    # 最小词频
    workers=4,      # 并行训练的线程数
    sg=1,           # 使用Skip-gram模型
    epochs=20       # 训练轮数
)

# 保存模型
w2v_model.save("model\\shakespeare_w2v.model")
# 输出模型信息
print("Word2Vec model trained and saved successfully.")
# 输出词向量示例
print("Example word vectors:")
for word in ["father", "mother", "man", "woman"]:
    if word in w2v_model.wv:
        print(f"{word}: {w2v_model.wv[word]}")
    else:
        print(f"{word} not found in the model vocabulary.")

# 尝试 father - man + woman ≈ mother
if all(word in w2v_model.wv for word in ["father", "man", "woman"]):
    result = w2v_model.wv.most_similar(positive=["father", "woman"], negative=["man"])
    print("father - man + woman 最相近的词：", result)
else:
    print("father、man、woman 其中有词不在词表中")
