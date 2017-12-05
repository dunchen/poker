import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np

length=34
info_dim=50+12
h_dim=12
z_dim=10
test_dim=50+6
testh_dim=10
action_dim=6

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

	def decoder(self,z):
	        outputs=[]
	        h_t2 = z
	        c_t2 = Variable(torch.zeros(1,testh_dim), requires_grad=False)
	        for i in self.test:
	                h_t2, c_t2 = self.decoder_lstm(i.unsqueeze(0), (h_t2, c_t2))
	                output= self.softmax(h_t2 @ self.Whx + self.bhx.repeat(h_t2.size(0), 1))
	                outputs += [output]
	        return outputs

		
	def forward(self,context,test):
		
                self.context=context
                self.test=test
	
                h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
                c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
                for i in self.context:
                    print(i)
                    h_t, c_t = self.encoder_lstm(i.unsqueeze(0), (h_t, c_t))
		
                z_mu=h_t @ self.Whz_mu + self.bhz_mu.repeat(h_t.size(0), 1)
                z_var = h_t @ self.Whz_var + self.bhz_var.repeat(h_t.size(0), 1)
                eps = Variable(torch.randn(1, z_dim))
                z = z_mu + torch.exp(z_var / 2) * eps
                
                print(z)
                pp=self.decoder(z)
                return pp,z_mu,z_var,z

def train():
        vae=VAE()
        vae_optimizer=optim.Adam(vae.parameters(),lr=1e-3)
        reconloss=nn.MSELoss()


        for i in range(casenum):
                vae_optimizer.zero_grad()
                context=infodata[i]
                test=testdata[i][0]
                target=testdata[i][1]
                outputs,z_mu,z_var,z=vae(context,test)
                klloss=torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
                print(outputs)
                print(target)
                loss=reconloss(torch.FloatTensor(outputs),target)+klloss
                loss.backward()
                vae_optimizer.step()
        return

def readinfo(x):
    out=[]
    for i in range(50):
        if i==6:
            out=out+[pd.to_numeric(x[i])]
        else:
            out=out+[x[i].astype(float)]
    return out

def readaction(x):
    out=[]
    for i in range(6):
        out=out+[x[i].astype(float)]
    return out

def add(x,y):
    out=[0 for col in range(50)]
    for i in range(50):
        out[i]=x[i].astype(float)+y[i].astype(float)
    return out

def checknotendround(x):
    return(pd.isnull(x[40]) and (x[0]!=-1))

fil=pd.read_csv('./2')
	
i=0
infodata=[]
testdata=[]
casenum=0
while i<length:
        i=i+3
        info=readinfo(fil.ix[i])
        i+=1
        info=add(info,readinfo(fil.ix[i]))
        i=i+6
	
        j=0
        numx=0
        numy=0
        inputx=[[0 for col in range(info_dim)] for roun in range(10)]
        inputy=[[0 for col in range(test_dim)] for roun in range(10)]
        target=[[0 for col in range(action_dim)] for roun in range(10)]

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

        infodata=infodata+[Variable(torch.FloatTensor(inputx))]
        testdata=testdata+[[Variable(torch.FloatTensor(inputy))]+[Variable(torch.FloatTensor(target))]]
        casenum=casenum+1
	
        if (fil.ix[i][0]==-1):
                i=i+1

        print(i)

train()
	
	




			
