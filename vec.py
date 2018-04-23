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



