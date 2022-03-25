import torch
import jieba
import os
import numpy as np

from Core.Datasets import MyDataset
from Core import CONFIG
from Core import Function
from Core.Model import GPT_Model


GPT = GPT_Model#模型
GPTconfig = CONFIG.GPTConfig#模型配置
Trainer = CONFIG.Trainer#模型训练器
Trainerconfig = CONFIG.TrainerConfig#训练配置
Sample = Function.sample#示例





#######功能实现代码：
train_name = str(input("\nplease inputs your datas:\n请输入您的要训练的数据文件名:"))
batch_size = int(input("\nplease inputs your batch_size:\n请输入您的要训练的batch_size这将取决于您显存的大小(如果您不确定请输入20):"))
epochs = int(input("\nEpochs:"))
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

bar = "=="
print("{}START TRAIN{}".format(bar*19,bar*19))

# 构建一个训练器
tconf = Trainerconfig(max_epochs=epochs, batch_size=batch_size)
trainer = Trainer(model, train_dataset, test_dataset=None, config=tconf, Save_Model_path='C:\\Users\\xbj0916\\Desktop\\M')
trainer.train()



