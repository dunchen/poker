import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np

nnum=4
info_dim=50+12
h_dim=12
z_dim=6
test_dim=50+6
testh_dim=6
action_dim=6
#path='~/Downloads/Final_fo_realz.txt'
path='./3'

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
                h_t2 = torch.tanh(z)
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

vae=VAE()
vae.load_state_dict(torch.load('./3000'))

def readinfo(x):
    out=[]
    for i in range(50):
        out=out+[float(pd.to_numeric(x[i]))]
    return out

def readaction(x):
    out=[]
    for i in range(6):
        out=out+[x[i].astype(float)]
    return out

def add(x,y):
    out=[0 for col in range(50)]
    for i in range(50):
        out[i]=x[i]+y[i]
    return out

def checknotendround(x):
    return(pd.isnull(x[40]) and (x[0]!=-1))

fil=pd.read_csv(path)
	
i=0
testnum=[0 for col in range(nnum)]
infodata=[]
testdata=[]
casenum=0
while (fil.ix[i,0]!=-1) and (casenum<nnum):
        i=i+3
        info=readinfo(fil.ix[i])
        i+=1
        info=add(info,readinfo(fil.ix[i]))
        i=i+6
	
        j=0
        numx=0
        numy=0
        inputx=[[-1 for col in range(info_dim)] for roun in range(10)]
        inputy=[[-1 for col in range(test_dim)] for roun in range(10)]
        target=[[-1 for col in range(action_dim)] for roun in range(10)]
        while checknotendround(fil.ix[i]):
                if ((j%2)==0):
                        inputy[numy]=info+readaction(fil.ix[i])
                        numy+=1
                else:
                        target[numy-1]=readaction(fil.ix[i])
                        #print(i,j,numy)
                        #print(target[numy-1])
                        
                        inputx[numx]=inputy[numy-1]+target[numy-1]
                        numx=numx+1
                j=j+1
                i=i+2
        print(i,fil.ix[i])
        if (j%2)!=0:
            numy=numy-1
        info=add(info,readinfo(fil.ix[i]))
        i+=1
        info=add(info,readinfo(fil.ix[i]))
        i+=1
        info=add(info,readinfo(fil.ix[i]))
        i+=1
        
        j=0
        while checknotendround(fil.ix[i]):
                if ((j%2)==0):
                        inputy[numy]=info+readaction(fil.ix[i])
                        numy+=1
                else:
                        target[numy-1]=readaction(fil.ix[i])
                        inputx[numx]=inputy[numy-1]+target[numy-1]
                        numx=numx+1

                j=j+1
                i=i+2

        print(i,fil.ix[i])
        if (j%2)!=0:
            numy=numy-1
        info=add(info,readinfo(fil.ix[i]))
        i+=1
	
        j=0     
        while checknotendround(fil.ix[i]):
                if ((j%2)==0):
                        inputy[numy]=info+readaction(fil.ix[i])
                        numy+=1
                else:
                        target[numy-1]=readaction(fil.ix[i])
                        inputx[numx]=inputy[numy-1]+target[numy-1]
                        numx=numx+1

                j=j+1
                i=i+2
	
        if (j%2)!=0:
            numy=numy-1
        info=add(info,readinfo(fil.ix[i]))
        i+=1

        j=0
        print('****',i,fil.ix[i])
        while checknotendround(fil.ix[i]):
                if ((j%2)==0):
                        inputy[numy]=info+readaction(fil.ix[i])
                        numy+=1
                else:   
                        target[numy-1]=readaction(fil.ix[i])
                        inputx[numx]=inputy[numy-1]+target[numy-1]
                        numx=numx+1
                        
                j=j+1
                i=i+2
        if (j%2)!=0:
            numy=numy-1

        infodata=infodata+[Variable(torch.FloatTensor(inputx))]
        testdata=testdata+[[Variable(torch.FloatTensor(inputy))]+[Variable(torch.FloatTensor(target))]]
        #print(numy)
        #print(numx)
        testnum[casenum]=numy
        #print(testnum)
        casenum=casenum+1
	
        if (fil.ix[i][0]==-1):
                i=i+1
        print('oooo')
        

'''
for i in range(casenum):
                context=infodata[i]
                test=testdata[i][0]
                target=testdata[i][1]
                outputs,z_mu,z_var,z=vae(context,test,testnum[i])
                klloss=torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
                loss=reconloss(outputs,target[0:testnum[i]])+klloss
                print(loss)
                print(reconloss(outputs,target[0:testnum[i]]),klloss)
                print(outputs)
                print(target[0:testnum[i]])
                print(i)
'''

context=infodata[0]
test=testdata[0][0]
target=testdata[0][1]
l0=0
l1=0
l2=0
l3=0
for j in range(1000):
    _,_,_,z0=vae(infodata[0],testdata[0][0],testnum[0])
    _,_,_,z2=vae(infodata[2],testdata[2][0],testnum[2])
    _,_,_,z1=vae(infodata[1],testdata[1][0],testnum[1])
    _,_,_,z3=vae(infodata[3],testdata[3][0],testnum[3])
    for i in range(z_dim):
        l0=l0+(z0.data[0][i]-z3.data[0][i])*(z0.data[0][i]-z3.data[0][i])
    for i in range(z_dim):
        l1=l1+(z1.data[0][i]-z3.data[0][i])*(z1.data[0][i]-z3.data[0][i])
    for i in range(z_dim):
        l2=l2+(z0.data[0][i]-z3.data[0][i])*(z0.data[0][i]-z3.data[0][i])
print(l0,l1,l2)

