import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from Core.Function import sample
from Core.Function import set_seed
from Core.CONFIG import Trainer
from Core.Model import GPT_Model
from Core.Function import kmeans
from Core.Datasets import ImageDataset
from Core.CONFIG import GPTConfig
from Core.CONFIG import TrainerConfig



#加载模型
path_name = str(input("Input the iGPT nodel name: "))
# 设置确定性
set_seed(42)


# ===========================下载数据====================================
# 加载数据
root = './'
train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
print(len(train_data), len(test_data))

# ================================================================
# 每张图像随机获取 5 个像素并将它们全部堆叠为 rgb 值以获得半百万个随机像素
pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()
print(px.size())

# ===========================应用K-means进行获取数据离散值=====================================
ncluster = 512
with torch.no_grad():
    C = kmeans(px, ncluster, niter=8)

print(C.size()) # 输出结果
# =============================制作数据集==============================
train_dataset = ImageDataset(train_data, C)                      # ==
#test_dataset = ImageDataset(test_data, C)                        # ==
print(train_dataset[0][0])  # 一个示例图像被展平为整数
                                                               # ==
# ===================================================================
# 训练前的一些GPT模型的配置
# 根据官方的模型，参数为batch_size = 128,Adam lr 0.003，beta = (0.9, 0.95)
# 学习率预热一个 epoch，然后衰减到 0
# 没有使用权重衰减或Droput
# n_layer=24, n_head=8, n_embd=512
# 另外您可以根据自己的设备进行自己配置
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=10, n_head=4, n_embd=84)

model = GPT_Model(mconf)
print(model)
# =============================Load Model=====================================

#path_iGPT = os.path.join('Pre_models',path_name)
checkpoint = torch.load('model.bin')
model.load_state_dict(checkpoint)


tokens_per_epoch = len(train_data) * train_dataset.block_size
train_epochs = 1 # todo run a bigger model and longer, this is tiny
# 初始化训练器进行训练
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=3*8, learning_rate=3e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      num_workers=8)

trainer = Trainer(model = model, train_dataset = train_dataset, test_dataset = None, config = tconf,Save_Model_path='./pa')
 

# =============================================================================


# to sample we also have to technically "train" a separate model for the first token in the sequence
# we are going to do so below simply by calculating and normalizing the histogram of the first token
counts = torch.ones(ncluster) # start counts as 1 not zero, this is called "smoothing"
rp = torch.randperm(len(train_dataset))
nest = 5000 # how many images to use for the estimation
for i in range(nest):
    a, _ = train_dataset[int(rp[i])]
    t = a[0].item() # index of first token in the sequence
    counts[t] += 1
prob = counts/counts.sum()


n_samples = 32
start_pixel = np.random.choice(np.arange(C.size(0)), size=(n_samples, 1), replace=True, p=prob)
start_pixel = torch.from_numpy(start_pixel).to(trainer.device)
pixels = sample(model, start_pixel, 32*32-1, temperature=1.0, sample=True, top_k=100)


# =========================Show the Images===============================
# for visualization we have to invert the permutation used to produce the pixels
iperm = torch.argsort(train_dataset.perm)

ncol = 8
nrow = n_samples // ncol
plt.figure(figsize=(16, 8))
for i in range(n_samples):
    pxi = pixels[i][iperm] # note: undo the encoding permutation
    
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(C[pxi].view(32, 32, 3).numpy().astype(np.uint8))
    plt.axis('off')
