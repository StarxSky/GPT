# 使用Pytorch构建的GPT-2训练中文语料
![pytorch logo](https://avatars.githubusercontent.com/u/21003710?s=88&v=4)

# 建议在Colab上使用GPU跑代码
![im](https://github.com/StarxSky/GPT-2/blob/main/%E7%AE%80%E4%BB%8B/pp.png?raw=true)

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
- 运行训练！
```
    >>> 进入CNGPT目录下
    >>> python train.py 
    
```
### 使用预训练的模型
- 下载预训练模型[Download](https://drive.google.com/file/d/133ERymhZejMj3aKwJLcLadMLUy0cw43w/view?usp=sharing)
- 您需要更改模型存放的位置路径，请参考以下代码：将您所下载的模型路径放到model_path中（文件为：pre.py)
```python
#模型的地址
model_path = '/content/drive/MyDrive/ColabNotebooks/CNGPT/Models/model.bin'
#训练数据的地址
path = '/content/drive/MyDrive/ColabNotebooks/CNGPT/datas/train.text'#linux


```

## 生成文章
![m](https://github.com/StarxSky/GPT-2/blob/main/%E7%AE%80%E4%BB%8B/h.png?raw=true)
```
    >>>生成文章
    >>>python pre.py
```
- 如果想要更改生成文章的字数可以修改pre.py中的"steps="

```python
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...] # context conditioning
y = Sample(model, x, steps=100, temperature=1.0, sample=True, top_k=10)[0]
print(y)
print('==============================DONE=================================')

```


### 生成文章时遇到的问题
- 当出现：RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.时尝试修改pre.py中的以下代码片段
- 原因是由于所下载的预训练模型使用CUDA训练的而您的设备不支持CUDA
```python
model.load_state_dict(torch.load(model_path,map_location='cpu'))
#当出现 RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.使用下方的方法将其添加到加载模型的语句中
#map_location=cpu

```
