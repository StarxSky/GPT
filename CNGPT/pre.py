import torch
import jieba
import os 
import numpy as np

from Dtasest import MyDataset
from Layers import Config
from Layers import utils
from Layers.Model import GPT_Model


GPT = GPT_Model#模型
GPTconfig = Config.GPTConfig#模型配置
Trainer = Config.Trainer#模型训练器
Trainerconfig = Config.TrainerConfig#训练配置
Sample = utils.sample#示例

#模型的地址
model_path = str(input("请输入预训练模型的名称在这之前请您确保下载了模型并且确保模型在Pre_models目录下："))
pre_model_path = os.path.join('Pre_models',model_path)

#训练数据的地址
train_name = str(input("\nplease inputs your datas:\n请输入您训练模型的数据:"))
# CUDA确认
GPU = bool(input("请确认您是否有CUDA(Yes:True,None:False)："))
# 分词
path_ = os.path.join('datas',train_name)
f = open(path_,encoding='utf-8').read()
aa = jieba.lcut(f)
print(aa)



# 构建 GPT 模型
train_dataset = MyDataset(aa,20)
mconf = GPTconfig(train_dataset.vocab_size,train_dataset.block_size, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = GPT(config = mconf)
print(model)

print("{}STARTN{}".format("=="*19,"=="*19))

if GPU == True :
  model.load_state_dict(torch.load(pre_model_path))
else :
  model.load_state_dict(torch.load(pre_model_path,map_location='cpu'))

print("Model was Load Done!")
#当出现 RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.使用下方的方法将其添加到加载模型的语句中
#map_location=cpu

# 构建一个训练器


# sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)
context = str(input('请给个文本:'))
steps = int(input("输入生成的字数："))
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...] # context conditioning
y = Sample(model, x, steps=steps, temperature=1.0, sample=True, top_k=10)[0]
print(y)
print('{}DONE{}'.format("=="*19,"=="*19))
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
