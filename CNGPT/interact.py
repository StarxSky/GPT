import os
import torch
import jieba
import numpy as np

from Core.Datasets import MyDataset
from Core import CONFIG
from Core import Function
from Core.Model import GPT_Model

GPT = GPT_Model  # 模型
GPTconfig = CONFIG.GPTConfig  # 模型配置
Trainer = CONFIG.Trainer  # 模型训练器
Trainerconfig = CONFIG.TrainerConfig  # 训练配置
Sample = Function.sample  # 示例

# 模型的地址
model_path = str(input("请输入预训练模型的名称在这之前请您确保下载了模型并且确保模型在Pre_models目录下："))
pre_model_path = os.path.join('Pre_models', model_path)
# CUDA确认
GPU = bool(input("请确认您是否有CUDA(Yes:True,None:False)："))

pp = str(input("请输入数据文件夹的名称："))


################

def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>吧一串字符串组合成路径
    return res


txts = []
for file in getFiles(pp, '.txt'):  # =>查找以.txt结尾的文件
    with open(file, "r", encoding='utf-8') as f:
        # 打开文件
        data = f.read()  # 读取文件
        txts.append(data)

f = ''.join(txts)  # 转化为非数组类型

# 分词
aa = jieba.lcut(f)
print(aa)

# 构建 GPT 模型
train_dataset = MyDataset(aa, 20)
mconf = GPTconfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=12, n_head=12, n_embd=768)  # a GPT-1
model = GPT(config=mconf)
print(model)

print("{}STARTN{}".format("==" * 19, "==" * 19))

if GPU == True:
    model.load_state_dict(torch.load(pre_model_path))
else:
    model.load_state_dict(torch.load(pre_model_path, map_location='cpu'))

print("Model was Load Done!")
# 当出现 RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.使用下方的方法将其添加到加载模型的语句中
# map_location=cpu

# 构建一个训练器


# sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)
steps = int(input("输入生成的字数："))
print('{}DONE{}'.format("==" * 19, "==" * 19))





history = []
while True:
    raw_text = str(input(">>> "))
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input(">>> ")
    raw_text = " ".join(list(raw_text.replace(" ", "")))

    x = torch.tensor([train_dataset.stoi[s] for s in raw_text], dtype=torch.long)[None, ...]
    history.append(x)

    y = Sample(model, x, steps=steps, temperature=1.0, sample=True, top_k=10)[0]
    history.append(y)

    out_text = ''.join([train_dataset.itos[int(i)] for i in y])
    print(out_text)

