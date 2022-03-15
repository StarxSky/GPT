# 使用Pytorch构建的GPT-2训练中文语料
![pytorch logo](https://avatars.githubusercontent.com/u/21003710?s=88&v=4)
## 使用方法
- 安装必要的包
- Tips1:
```
>>> pip install torch torchvision //windows
>>> pip install torch //Orther
>>> pip install jieba
>>> pip install tqdm

```
- Tips2(老少皆宜)

```
>>> pip install -r requirments.txt
```


- 在这之前您需要改动一下训练数据的存放地址 path

```python
# 分词
path = 'datas/train.text'#linux
path = 'datas\\train.text'#windows
f = open(path,encoding='utf-8').read()
aa = jieba.lcut(f)
print(aa)

```
- 这里的路径用来指定文件夹存放log
```python
# 构建一个训练器
tconf = Trainerconfig(max_epochs=1, batch_size=256)
trainer = Trainer(model, train_dataset, test_dataset=None, config=tconf, Save_Model_path='C:\\Users\\xbj0916\\Desktop\\M')
trainer.train()
```
- 运行！
```
    >>> 进入CNGPT目录下
    >>> python train.py 
```
