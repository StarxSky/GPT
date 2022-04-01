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


pp = str(input("输入您存放训练数据的文件夹目录："))
################
# 得到文本文件
def getFiles(dir, suffix): # 查找根目录，文件后缀 
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
    return res



  
txts = []
for file in getFiles("datas", '.txt'):  # =>查找以.txt结尾的文件
       with open(file, "r",encoding='utf-8') as f: 
            
            #打开文件
            data = f.read()   #读取文件
            txts.append(data)
        
f = ''.join(txts)#转化为非数组类型 



#######功能实现代码：

batch_size = int(input("\nplease inputs your batch_size:\n请输入您的要训练的batch_size这将取决于您显存的大小(如果您不确定请输入20):"))
epochs = int(input("\nEpochs:"))
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



