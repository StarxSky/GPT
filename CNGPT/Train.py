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





#######功能实现代码：
train_name = str(input("\nplease inputs your datas:\n请输入您的要训练的数据文件名:"))
batch_size = str(input("\nplease inputs your batch_size:\n请输入您的要训练的batch_size这将取决于您数据集的大小(如果您不确定请输入20):"))
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
tconf = Trainerconfig(max_epochs=20, batch_size=batch_size)
trainer = Trainer(model, train_dataset, test_dataset=None, config=tconf, Save_Model_path='C:\\Users\\xbj0916\\Desktop\\M')
trainer.train()

# (... enjoy the show for a while... )



# sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)
#x = torch.tensor([1, 2, 3], dtype=torch.long)[None, ...] # 语境条件反射
#y = Sample(model, x, steps=30, temperature=1.0, sample=True, top_k=5)[0]
#print(y) # 我们的模型用 30 个额外的可能整数填充整数序列
