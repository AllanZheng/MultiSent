import numpy as np
import os
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print(syn1)
print(syn0)
print(l2)


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