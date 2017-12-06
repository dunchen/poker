import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np

nnum=10000
info_dim=50+12
h_dim=12
z_dim=10
test_dim=50+6
testh_dim=10
action_dim=6
path='~/Downloads/Final_fo_realz.txt'
#path='./2'

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.randn(*size) * xavier_stddev

class VAE(nn.Module):
        def __init__(self):
                super(VAE, self).__init__()
                self.encoder_lstm=nn.LSTMCell(info_dim,h_dim)
                self.Whz_mu = nn.Parameter(xavier_init(size=[h_dim, z_dim]))
                self.bhz_mu = nn.Parameter(torch.zeros(z_dim))
                self.Whz_var = nn.Parameter(xavier_init(size=[h_dim, z_dim]))
                self.bhz_var = nn.Parameter(torch.zeros(z_dim))
                self.softmax=nn.Softmax()
                self.decoder_lstm=nn.LSTMCell(test_dim,testh_dim)
                self.Whx = nn.Parameter(xavier_init(size=[testh_dim, action_dim]))
                self.bhx = nn.Parameter(torch.zeros(action_dim))

                self.context=[]
                self.test=[]

        def decoder(self,z,num):
                outputs=[]
                h_t2 = z
                c_t2 = Variable(torch.zeros(1,testh_dim), requires_grad=False)
                for i in range(num):
                        h_t2, c_t2 = self.decoder_lstm(self.test[i].unsqueeze(0), (h_t2, c_t2))
                        output= self.softmax(h_t2 @ self.Whx + self.bhx.repeat(h_t2.size(0), 1))
                        outputs += [output]
                return torch.stack(outputs,1).squeeze()


        def forward(self,context,test,num):

                self.context=context
                self.test=test

                h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
                c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
                for i in range(num):
                    h_t, c_t = self.encoder_lstm(self.context[i].unsqueeze(0), (h_t, c_t))

                z_mu=h_t @ self.Whz_mu + self.bhz_mu.repeat(h_t.size(0), 1)
                z_var = h_t @ self.Whz_var + self.bhz_var.repeat(h_t.size(0), 1)
                eps = Variable(torch.randn(1, z_dim))
                z = z_mu + torch.exp(z_var / 2) * eps

                pp=self.decoder(z,num)
                return pp,z_mu,z_var,z
