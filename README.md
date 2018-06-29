# Multilingual sentence classification 多语种文本分类

This is the implmentation of using [Facebook Multilingual word embeddings](https://github.com/facebookresearch/MUSE)  to support the model on using to predict the multilingual sentence.
The classification algroithm is based on the [ Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
and it is a simple neural network but useful and effective.



这是基于Facebook 多语种的词向量去实现模型基于单语言数据集进行不同于训练语言的文本分类。
当前目标的语言包含英语，法语和阿拉伯语,当前Facebook提供了30种语言在同一空间下的词向量，如果新增的目标语言不包含在Facebook里，可以调用Facebook提供的程序[Facebook Multilingual word embeddings](https://github.com/facebookresearch/MUSE) 与Facebook的词向量进行对齐训练。

## Sepcification
>  run.py 主程序，包含train()训练模型，test()测试函数，pre_process()导入词向量，similar()寻找相似词, predict（）预测函数（待完善）

> model.py 算法模型

>  utils.py 提供数据预处理和读取，模型读取和存储函数

## Execution 运行方式:

> python3 run.py --help

```


usage: run.py [-h][--mode MODE] [--model MODEL ][--dataset DATASET ]
			  [ --save   model ][ --early_stopping] [--epoch EPOCH]
			  [--learning_rate LEARNING_RATE][--gpu GPU] [--lang  LANG][--simIlar SIMILAR]

-----[CNN-classifier]-----

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           train: train (with test) a model / test: test saved
						models/ predict: saved mdoels to predict (to be continued)
  --model MODEL         available models: 0(not used word embeddings),1(used word embeddings)
  --dataset DATASET     available datasets: MR,M
  --save_model          whether saving model or not
  --early_stopping      whether to apply early stopping
  --epoch EPOCH         number of max epoch
  --learning_rate LEARNING_RATE
						learning rate
  --gpu GPU             the number of gpu to be used
  --lang                the predict language used 
  --similar            whether used the similar function
```




## Data format 数据格式 

数据文本样式以__label__分割，前面是文本，后面是类别

The data format is shown on below:

Je me suis rendu compte que je n'ai plus 38728 pour l'essence ce mois-ci !! J'ai besoin de faire 1/4 de réservoir les 2 dernières semaines`__label__`negative




## Development Environment 环境配置

- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- GPU: GTX 1080

## Dependencies 依赖库



- diiflib==2.4.0
- numpy==1.12.1
- gensim==2.3.0
- scikit_learn==0.19.0



