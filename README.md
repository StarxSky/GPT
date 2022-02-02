# TF2_GPT-2
```
   █████████████████████        █████████████████████        █████████████████████
   █████████████████████        █████████████████████          █████████████████
   ███████                      ███████       ███████               ███████
   ███████    ██████████        ███████       ███████               ███████
   ███████      ████████        █████████████████████               ███████
   ███████          ████        █████████████████████               ███████
   █████████████████████        ███████                             ███████
   █████████████████████        ███████                             ███████
```

- Use Tensorflow2.7.0 Build OpenAI'GPT-2
- 使用最新tensorflow2.7.0构建openai官方的GPT-2 NLP模型
# Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Xhs753/TF2_GPT-2/HEAD)

## 优点

- 使用无监督技术
- 拥有大量词汇量
- 可实现续写
- 实现对话后续将应用于FloatTech的Bot

## 食用方法

### Setting

*  python >= 3.6
*  numpy==1.16.4
*  sentencepiece==0.1.83
*  tensorflow-gpu==2.7.0

#### Steps

```
1. git clone https://github.com/Xhs753/TF2_GPT-2
2. $ cd TF2_GPT-2
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


###### 在任意数据上训练

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
![/image](https://github.com/Xhs753/TF2_GPT-2/blob/main/%E7%AE%80%E4%BB%8B/GPT-2_Model.jpg)


# Link
- [OpenAi-GPT-2](https://github.com/openai/gpt-2)


# Thanks To My Friends 
- [FloatTech](https://github.com/FloatTech)
- [夜黎](https://github.com/DawnNights)
- [MayuriNFC](https://github.com/MayuriNFC)
- [理酱](https://github.com/Yiwen-Chan)



# LICENCE

```
███████   ███   ███████       ███████        ██████████████████████████████████
███████   ███   ███████                      ██████████████████████████████████        
███████   ███   ███████       ███████                    ██████████
███████   ███   ███████       ███████                    ██████████
███████   ███   ███████       ███████                    ██████████
███████   ███   ███████       ███████                    ██████████
███████   ███   ███████       ███████                    ██████████
███████   ███   ███████       ███████                    ██████████

```
- [MIT](https://github.com/Xhs753/TF2_GPT-2/blob/main/LICENSE)


