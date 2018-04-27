# -*- coding: utf-8 -*-
import nltk as
import io
import re
# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchwordemb
from Network import RNN
def pre_vector():
    vocab, vec = torchwordemb.load_word2vec_text("vec/wiki.multi.fr.vec")
    vocab1, vec1= torchwordemb.load_word2vec_text("vec/wiki.multi.ar.vec")
    x=torch.cat((vec,vec1))
    vocab.update(vocab1)
    return vec,vocab

print(x.size())
print(x[vocab["تصنيف"]])

    #raw = f.readlines()

def pre_Process(content):
    content=content.lower()
   v
    label=content.split(" ")[0]
    content=content.split(" ")
    return label,content

def w2v(vec,vocab,content):
    for word in content:
        if(word in vocab):
            word=vec
            #transfer the word into the corresponding  vector
