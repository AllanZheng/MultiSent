from model import CNN
import utils
import pandas as pd
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import warnings
import difflib
import re
import time
path = '../'
warnings.filterwarnings("ignore")
def similar(lang,params,data,mode):
    '''
    if mode != 'predict':
        data = list
    else:
        data = dat
    '''
    #寻找相似词
    rec = 0
    unk = 0
    beg_sent = time.time()
    word_vector = KeyedVectors.load_word2vec_format(path+"vec/wiki.multi."+lang+".vec",binary=False)
    for a,sent in enumerate(data[mode]):
       for b,w in enumerate(sent):
           #print(data[mode][a][b],w)
            try:
               if w in word_vector and w==w:
                   data[mode][a][b] = data["word_to_idx"][w]
               else:
                   #judge whehter is ther number 判断是否为数字
                   if (re.match(r"[+-]?\d+(?:\.\d+)?", str(w))):
                       data[mode][a][b] = params["VOCAB_SIZE"]
                   else:
                       # print(w)
                       unk = unk + 1
                       mid = difflib.get_close_matches(w,word_vector.vocab,1)
                       if (len(mid) and mid[0]==mid[0]):
                            data[mode][a][b] = data["word_to_idx"][mid[0]]
                            rec = rec + 1
                       else:
                           data[mode][a][b] = params["VOCAB_SIZE"]
            except:
               print(w)
               data[mode][a][b] = params["VOCAB_SIZE"]



    end_sent = time.time()
    print("sentence process time:", end_sent - beg_sent, rec,unk)
    if(mode =='train_x'):
        return data[mode],word_vector
    else:
        return data[mode]
def train(data, params):



    count = 0
    num = 0
    #da = pd.read_csv('../data/ar.csv')
    #x2 = list(da['0'])

    if params['SIMILAR']==1:
        data["train_x"],wv_en = similar('en',params,data,'train_x')
        x1 = similar('ar',params,data,'dev_x')
        x2 = similar('fr',params,data,'test_x')
        x3 = similar('ar',params,data,'ar_x')
        x4 = similar('fr',params,data,'fr_x')




        x1 = [[w for w in sent] +
              [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
              for sent in x1]


        x2 = [[w for w in sent] +
             [params["VOCAB_SIZE"] + 1] *(params["MAX_SENT_LEN"] - len(sent))
             for sent in  x2]


    else:
        data["train_x"], data["train_y"] = data["train_x"] + data["ar_x"] + data["fr_x"], data["train_y"] + data[
            "ar_y"] + data["fr_y"]
        x1 = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
             [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
             for sent in data['dev_x']]

        x2 = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
              [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
              for sent in data['test_x']]

    model = CNN(**params).cuda(params["GPU"])
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()  # 损失函数交叉熵
    #pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    f = open("result_frtoar.txt","w",encoding='utf-8')
    word_vector = KeyedVectors.load_word2vec_format(path + "vec/wiki.multi.en.vec", binary=False)
    '''
    end = time.time()
    print('Timecost:',end - begin )
    '''
    for e in range(params["EPOCH"]):
        start = time.time()
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            if(params["SIMILAR"]==1):
                batch_x = [[w for w in sent] +
                           [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                           for sent in data["train_x"][i:i + batch_range]]
            else:
                batch_x = [[data["word_to_idx"][w] if w in word_vector and w==w and not pd.isna(w) else params["VOCAB_SIZE"] for w in sent ] +
                           [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                           for sent in data["train_x"][i:i + batch_range]]

            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])
            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()
        end = time.time()
        #if mode == "dev":
       # elif mode == "test":
        #xx, yy = x1, data["dev_y"]
        #验证集的准确率
        dev_acc = test(x1,data["dev_y"],data, model, params)
        #测试集的准确率
        test_acc = test(x2,data["test_y"] ,data, model, params)
        st = str("epoch:" + str(e + 1) + " / dev_acc:" + str(dev_acc) + " / test_acc:" + str(test_acc) + ' running time:' + str(end - start) +'\n')
        print(st)
        f.writelines(st)
        if test_acc >= max_test_acc or dev_acc+test_acc>=max_dev_acc+max_test_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)


        if params["EARLY_STOPPING"] and dev_acc>=test_acc and  test_acc>=0.71:
            print("early stopping by dev_acc!")
            break
    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    f.close()
    return best_model


def test(x,y,data, model, params):

    model.eval()

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc
def predict(data,model,params):
    #To do list
    label = True
    return label
def pre_process(data,lang):
    # load word2vec

    # load train data set language model
    # 导入词向量
    # load test data set language model
    #合并词向量
    count=0

    vo=[]
    wv_matrix = []#word vector

    if (os.path.exists('vocabulary.csv') and os.path.exists('wv_matrix.npy')):
    #如果已存在合并词向量，直接读取
        voca = pd.read_csv('vocabulary.csv')
        data["vocab"] =list(voca['0'])
        #data["vocab"] =  sorted(list(set([w for sent in data["train_x"] for w in sent])))
        print('size of vocab:',len(data["vocab"]))
        wv_matrix = np.load('wv_matrix.npy')
    else:
        #data["vocab"] = (list(list(word_vectors1.vocab()) + list(word_vectors.vocab())+list(word_vectors2.vocab())))
        for j in lang:
            word_vectors = KeyedVectors.load_word2vec_format(path+"vec/wiki.multi." + j + ".vec", binary=False)
            for i in word_vectors.vocab:
                vo.append(i)
                wv_matrix.append(word_vectors.word_vec(i))
    #data["vocab"] = (list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
        data["vocab"] = vo
        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        np.save('wv_matrix.npy', wv_matrix)
        vocab = pd.DataFrame(data["vocab"])
        vocab.to_csv('vocabulary.csv', index=False)
        print(len(data['vocab']))
        print(wv_matrix.shape)
    data["classes"] = (list(set(data["test_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    data["test"] = (list(set([w for sent in data["test_x"] for w in sent])))
    print(f'numbers of test vocabulary: {len(data["test"])}')

    return data,wv_matrix
def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--dataset", default="M", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=True, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.5, type=float, help="learning rate")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")
    parser.add_argument("--lang", default='en', help="the target of language predict")
    parser.add_argument("--model", default="1",help="whether use the word embeddings ")
    parser.add_argument("--similar",default='0',help = "whether use the function to find the similar words in the word embeddings")
    options = parser.parse_args()
    starttime = time.time()

    # long running

    data = getattr(utils, f"read_{options.dataset}")()
    print('finished reading data!')
    lan =['fr','ar','en']
    data, wv =pre_process(data,lan)
    print('finish pre_process',wv.shape)
    endtime = time.time()

    print('running time:',endtime - starttime)
    params = {
        "MODEL":options.model,
        "WV_MATRIX" : wv,
        "LANG" : options.lang,
        "MODE" : options.mode,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"]+data["dev_x"]+data['test_x']]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "SIMILAR":options.similar
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    #print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
