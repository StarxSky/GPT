# 适合小白的教程


- 本教程只展示了使用预先训练的数据生成文本
- 此GPT由Tensorflow2构建只能训练英文语料！
- 如果想训练中文GPT-2请您移步至[CNGPT](https://github.com/StarxSky/GPT/tree/main/CNGPT)
- 本教程不包括[CNGPT](https://github.com/StarxSky/GPT/tree/main/CNGPT)

## steps :

 ```
1. git clone https://github.com/Xhs753/GPT
2. $ cd GPT/TF_GPT
3. $ pip install -r requirments.txt
4. $ python pre_process.py
5. $ python train_gpt2.py
6. $ python sequence_generator.py
```

### 对于有经验的用户

#### Steps

```
1. git clone https://github.com/StarxSky/GPT
2. $ cd GPT
3. $ pip install -r requirments.txt

```

- 你可以使用词仓库提供的sample.py示例数据预训练模型
#####　对仓库的可用数据进行训练模型

```
$ pyton pre_process.py --help

可选项：
  --data-dir TEXT        训练数据路径  [默认: /data/scraped]
  --vocab-size INTEGER   词汇大小和字节大小  [默认: 24512]
  --min-seq-len INTEGER  最小词序长度  [默认: 15]
  --max-seq-len INTEGER  最大词序sequence长度  [默认: 512]
  --help                 显示所有信息并退出
  
  
 ==>>python pre_process.py

```


##### 在任意数据上训练

```
>> python pre_process.py --data-dir=data_directory --vocab-size=32000

```

- 有关模型的命令源码在此
```
@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=24512, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=8, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--graph-mode', type=bool, default=False, show_default=False, help="TF run mode")
@click.option('--distributed', type=bool, default=False, show_default=False, help="distributed training")

```

# 使用GPT-2

```
>> python train_gpt2.py \
  --num-layers=8 \
  --num-heads=8 \
  --dff=3072 \
  --embedding-size=768 \
  --batch-size=32 \
  --learning-rate=5e-5
  --graph-mode=True
```






# 模型架构
![/image](https://github.com/StarxSky/GPT-2/blob/main/%E7%AE%80%E4%BB%8B/GPT-2_Model.jpg)


