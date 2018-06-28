# Multilingual sentence classification 多语种文本分类

这是基于Facebook 多语种的词向量去实现算法模型基于单向量
This is the implmentation of using [Facebook Multilingual word embeddings](https://github.com/facebookresearch/MUSE)  to support the model on using to predict the multilingual sentence.
The classification algroithm is based on the [ Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
and it is a simple neural network but useful and effective 
The re

When it is used by 
目前的做法是将Facebook
新增的目标语言
## Excution:

'''
python3 run.py --help
usage: run.py [-h] [--mode MODE] [--model MODEL] [--dataset DATASET]
			  [--save_model] [--early_stopping] [--epoch EPOCH]
			  [--learning_rate LEARNING_RATE] [--gpu GPU] [--lang ] [--simliar ]

-----[CNN-classifier]-----

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           train: train (with test) a model / test: test saved
						models/ predict: 
  --model MODEL         available models: rand, static, non-static,
						multichannel
  --dataset DATASET     available datasets: MR,
  --save_model          whether saving model or not
  --early_stopping      whether to apply early stopping
  --epoch EPOCH         number of max epoch
  --learning_rate LEARNING_RATE
						learning rate
  --gpu GPU             the number of gpu to be used
  --lang 
  --similiar 
'''

The data format is shown on below:
数据文本样式以__label__分割，前面是文本，后面是类别
'''

'''
## Development Environment 环境配置

- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- GPU: GTX 1080

## Dependencies 依赖

'''
diiflib==2.4.0
numpy==1.12.1
gensim==2.3.0
scikit_learn==0.19.0
'''

##

## Reference

If you use the code in your project, please consider mentioning in your README.

If you use the code in your research, please consider citing the library as follows:
'''
@misc{Multilingual sentence classification,
  author = {Qiaoyang Zheng}
  title = {Multilingual Sentence Classification},
  year = {2018},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Tencent/PhoenixGo}}
}
'''
### 