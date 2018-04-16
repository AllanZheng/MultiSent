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
        hidden = self.i2h(combined) #N*hidden_size，这里计算了一个hidden，hidden会带来下一个combined里
        output = self.i2o(combined) # N*output_size,就是一个普通全连接层
        output = self.softmax(output)#softmax
        return output, hidden

   def initHidden(self):

        return Variable(torch.zeros(1, self.hidden_size))#hidden=[1,hidden_size]

n_hidden = 128
target_size = 10#分类问题，分成几类
rnn = RNN(26, n_hidden, target_size)
hidden = rnn.initHidden()
#如果输入input，是一个char，按one-hot编码的，假设char空间是26（小写英文字母）
input = Variable(torch.zeros((1,26)))#[torch.FloatTensor of size 1x26],成员都是0
print(input)
output, next_hidden = rnn(input, hidden)#得到[1*10,1*128]
print(output.data.size(),next_hidden.data.size())

