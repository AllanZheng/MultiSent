# -*- coding: utf-8 -*-
import numpy as np
import word2vec
import torch
import torchvision
import torchvison.transforms as transforms
class Net(nn.Modules):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward:
        if(i==0):
            while(i<0):
               if(x>0)




def w2c():
    # we give an example of this function in the day 1, word vector notebook
    word_to_index, word_vectors, word_vector_size = load_word_vectors()


    # now, we want to iterate over our vocabulary items
    for word, emb_index in vectorizer.word_vocab.items():
    # if the word is in the loaded glove vectors
        if word.lower() in word_to_index:
         # get the index into the glove vectors
         glove_index = word_to_index[word.lower()]
         # get the glove vector itself and convert to pytorch structure
         glove_vec = torch.FloatTensor(word_vectors[glove_index])

         # this only matters if using cuda :)
         if settings.CUDA:
             glove_vec = glove_vec.cuda()

         # finally, if net is our network, and emb is the embedding layer:
         net.emb.weight.data[emb_index, :].set_(glove_vec)