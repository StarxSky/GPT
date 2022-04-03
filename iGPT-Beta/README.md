# iGPT - Beta
- 顾名思义这是个iGPT的测试版本皆旨在开发iGPT的新功能和测试其的性能，并进行优化和升级
- 难免会出现Bug，还望大家在ISSUES上发表您的意见，我会认真改正！！

# 如何食用？How To Use?
## 1.Step:（```Train.py```文件）
- 请将这里的```pt_datast_path```改为您的数据集存放的路径（最好放到```iGPT```的目录下，并创建一个名为```datas```的文件夹用来存放数据）
```python
# ==============================使用本地数据集========================================
train_data = Images_load_local(pt_dataset_path = 'datas')
print(len(train_data)) #len(test_data))
```

## 2.Step:(```Train.py```文件)
- 可以的话您可以适当的调整```Batch_size```的大小以来优化您的模型训练的速度，减少内存的占用

```python
if __name__ == '__main__':

    tokens_per_epoch = len(train_data) * train_dataset.block_size
    train_epochs = 1 # todo run a bigger model and longer, this is tiny
# 初始化训练器进行训练
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=3*8, learning_rate=3e-3,
                        betas = (0.9, 0.95), weight_decay=0,
                        lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                        num_workers=8)

    trainer = Trainer(model = model, train_dataset = train_dataset, test_dataset = None, config = tconf,Save_Model_path='./pa')
    trainer.train()
```


## 3.Step:（```Train.py```文件）
- 请您根据您的数据集大小进行调整集群参数```ncluster```
- 您可以根据这个公式进行调整```ncluster - 1 = (您数据集的大小)```

```python
# ===========================应用K-means进行获取数据离散值=====================================
ncluster = 106
#出现问题：RuntimeError: shape mismatch: value tensor of shape [105, 3]  （ncluster - 1 = value tensor shape）
# cannot be broadcast to indexing result of shape [106, 3]，应该调整这里的参数为正确值！
with torch.no_grad():
    C = kmeans(px, ncluster, niter=8)
```

## 4.Step:(Core核心的```FUNCTION.py```文件)
- 如果遇到问题在```k-means```聚类函数上请参考一下代码进行Debug!
```python
# ==============================================
# 编写k-means函数
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # 随机初始化数据集群
    for i in range(niter):

        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1) # 将所有像素分配给最近的标本元素
        # 将每个码本元素移动为分配给它的像素的平均值
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        #重新分配 任何位置不佳的码本元素
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        # ============Debug=============
        #print(c.shape)
        #print("========================")
        #print(ndead)
        #print("========================")
        #print(nanix)
        # ==============================
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c
    
  ```
  
  # [LICENSE](https://github.com/StarxSky/GPT/blob/main/LICENSE)
