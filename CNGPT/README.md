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


- 在这之前您需要预先将所要训练的数据提前放到```datas```目录下

```python
#######功能实现代码：
train_name = str(input("\nplease inputs your datas:\n请输入您的要训练的数据:"))

# 分词
path_ = os.path.join('datas',train_name)

f = open(path_,encoding='utf-8').read()
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
- 进行微调
  - ```block_size``` :只能关注前面词的数量
```python
# 构建 GPT 模型
train_dataset = MyDataset(aa,20)block_size = 20 
mconf = GPTconfig(train_dataset.vocab_size,train_dataset.block_size, n_layer=12, n_head=12, n_embd=768) # a GPT-1
model = GPT(config = mconf)
print(model)
```
- 运行训练！
```
    >>> 进入CNGPT目录下
    >>> python Train.py 
    
```
### 使用预训练的模型
|下载预训练模型
|------------------
| [DOWNLOAD 370MB](https://drive.google.com/file/d/133ERymhZejMj3aKwJLcLadMLUy0cw43w/view?usp=sharing)
| [DOWNLOAD 2.43GB](https://drive.google.com/file/d/1WyzkpDFlztRrG9nHqW0W1A29bX7VjIJM/view?usp=sharing)

- 您需要将所下载的预训练模型或者已训练好的模型提前放置在```Pre_models```目录下
- 注意！！您用哪种文本语料训练的CNGPT您就需要把您的语料路径填写进去！！(默认的语料库是```datas```中的```train.text```，因此，如果您确定默认的话请您将训练的数据填写```train.text```)

```python
#模型的地址
model_path = str(input("请输入预训练模型的名称在这之前请您确保下载了模型并且确保模型在Pre_models目录下："))
pre_model_path = os.path.join('Pre_models',model_path)

#训练数据的地址
train_name = str(input("\nplease inputs your datas:\n请输入您的要训练的数据:"))
path_ = os.path.join('datas',train_name)
```
# 交互模式
```
 >>> python interact.py
```
# 网页交互模式
- 直接运行```webapp.py```即可。
  - Dash application: A nice web interface for the dialogue model (Author: [xiejiachen](https://github.com/xiejiachen))


# 生成文章
![m](https://github.com/StarxSky/GPT-2/blob/main/%E7%AE%80%E4%BB%8B/h.png?raw=true)
```
    >>>生成文章
    >>>python pre.py
```


# 生成文章时遇到的问题
- 当出现：RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
- 原因是由于所下载的预训练模型使用CUDA训练的而您的设备不支持CUDA
```python
# CUDA确认
GPU = bool(input("请确认您是否有CUDA(Yes:True,None:False)："))
if GPU == True :
  model.load_state_dict(torch.load(pre_model_path))
else :
  model.load_state_dict(torch.load(pre_model_path,map_location='cpu'))
```

### 如果报错key"xxx"

- 这是由于数据集中没有这个词，请尝试将以下分词代码关掉，也就是直接将文本原始数据输入，不进行分词
- 请尝试更换数据集或者更换一个您数据集中含有的词汇。

```python
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
```
