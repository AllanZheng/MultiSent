import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
#这里自己实现一个RNN.
   #input_size就是char_vacab_size=26,hidden_size随意，就是隐层神经元数，output_size要分成categories类
   def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()#初始化类

        self.hidden_size = hidden_size#隐藏层的层数

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

   def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)#input:[N,26]+[N,hidden_size]=N,26+hidden_size -> N*hidden_size
        w
        hidden = self.i2h(combined) #N*hidden_size，这里计算了一个hidden，hidden会带来下一个combined里
        output = self.i2o(combined) # N*output_size,就是一个普通全连接层
        output = self.softmax(output)#softmax
        return output, hidden

   def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))#hidden=[1,hidden_size]

learning_rate = 0.005
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def  Train_process():
    return 0


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)