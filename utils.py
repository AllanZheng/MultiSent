from sklearn.utils import shuffle
import string
import pickle
import re
import requests
import json
import os
#from googletrans import Translatorw 文本分词操作，去除标点符号
path = '../'
def clean(line):
    #数据清洗
    line = re.sub(r"http\S+", "", line, flags=re.I)
    line = re.sub(r"@\S+", "", line, flags=re.I)
    line = re.sub(r"RT", "", line, flags=re.I)
    line = line.lower()
    line = line.replace('-', ' ')
    #line = re.sub("[\s+\.\!\/_,$%^*(+\"\'?!\[\]+|[+——！，。？、~@#￥%……&*（）]+", " ", line)
    remove = string.punctuation
    remove = remove.replace("'", "")
    line = "".join((char for char in line if char not in remove))
    #line ="".join((char for char in line if char not in string.punctuation))
    return line

def send_request(St):
    # Algo Classify
    # POST http://algo2:20990/classify/classifyOne
    #分词
    try:
        response = requests.post(
            url="http://algo2:20990/classify/classifyOne",
            headers={
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "data": {
                    "Text": St
                }
            })
        )
        content=str(response.content.decode())

        content = content.split('_#2DEL2#_POS_#1DEL1#_')[0]
        try:
            content = content.split('__#2DEL2#_SEG_#1DEL1#_')[1]
        except:
            content =''
        return content

    except requests.exceptions.RequestException:
        print('HTTP Request failed')
        os._exit()

def read_MR():

    x, y = [], []

    with open(path+"data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append('positive')

    with open(path+"data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append('negative')

    x, y = shuffle(x, y)


    return x,y

def read_M():
    data = {}

    def read(mode):
        x, y = [], []
        pos=0
        neg=0
        net=0
        count=0
        #trans = Translator()
        x1, y1 = [], []
        x, y = read_MR()
        dev_idx = len(x)//10

        #data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
        #data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        data["train_x"], data["train_y"] = x, y
        x, y = [], []
           


        with open(path+"data/" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:

                if line[-1] == "\n":
                    line = line[:-1]
                if line.split("__label__")[1] == 'positive':
                    pos = pos + 1


                elif line.split("__label__")[1] == 'negative':
                    neg = neg + 1
                elif (line.split("__label__")[1] == 'netural'):
                    net = net + 1
                    #continue
                #y.append(line.split("__label__")[1])
                label = line.split("__label__")[1]
                line = line.split("__label__")[0]
                line = clean(line)

                mid = send_request(line)
                if(mid):

                   count = count + 1
                   line =mid

                if net<300 and label =='netural':
                    x1.append(line.split())
                    y1.append(label)
                
                else:

                    x.append(line.split())
                    y.append(label)
        #x, y = shuffle(x, y)

            
        if mode == "train":
            x, y = shuffle(x, y)

            #x1 = len(x)//10
            #dev_idx = len(x)//10
            data["dev_x"], data["dev_y"] = x,y
            data["ar_x"],data["ar_y"]=x1,y1
            #data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]


        else:
           
            #x1 = len(x)//20

            x, y = shuffle(x, y)

            data["test_x"], data["test_y"] = x,y
            data["fr_x"], data["fr_y"] = x1, y1
        print(neg, pos,net)
        total = net+pos+neg
        print(count,total)
        '''
        with open('../data/'+mode+'data.txt','w',encoding='utf-8') as f:
            f.write(x)
        with open('../data/'+mode+'label.txt','w',encoding='utf-8') as f:
            f.write(y)
        '''
    read("train")
    read("test")

    return data

def save_model(model, params):
    #path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    path = f"saved_models/{params['DATASET']}_{params['EPOCH']}_{params['LANG']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['EPOCH']}_{params['LANG']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
