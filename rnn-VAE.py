import torch.nn as nn
from torch.autograd import Variable

length=3935699
info_dim=50+12
h_dim=12
z_dim=10
test_dim=50+6
testh_dim=10
action_dim=6

class VAE(nn.Module):
	def __init__(self):
		super(Sequence, self).__init__()
		self.encoder_lstm=nn.LSTMCell(info_dim,h_dim)
		self.Whz_mu = nn.Parameter(xavier_init(size=[h_dim, z_dim]))
                self.bhz_mu = nn.Parameter(torch.zeros(z_dim))
                self.Whz_var = nn.Parameter(xavier_init(size=[h_dim, z_dim]))
                self.bhz_var = nn.Parameter(torch.zeros(z_dim))
		
		self.decoder_lstm=nn.LSTMCell(test_dim,testh_dim)
                self.Whx = nn.Parameter(xavier_init(size=[testh_dim, action_dim]))
                self.bhx = nn.Parameter(torch.zeros(action_dim))
			
	def decoder(z)
		outputs=[]
		h_t2 = z
                c_t2 = Variable(torch.zeros(1,testh_dim), requires_grad=False)
                for i in self.inputy:
                        h_t2, c_t2 = self.decoder_lstm(i, (h_t2, c_t2))
                        output= nn.Softmax(h_t2 @ self.Whx + self.bhx.repeat(h_t2.size(0), 1))
                        ouputs += [output]
                return outputs

		
	def forward(self,context,test):
		
		self.context=context
		self.test=test
	
		h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
		c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
		for i in self.context:
			h_t, c_t = self.encoder_lstm(i, (h_t, c_t))
		
		z_mu=h_t @ self.Whz_mu + self.bhz_mu.repeat(h_t.size(0), 1)
		z_var = h_t @ self.Whz_var + self.bhz_var.repeat(h_t.size(0), 1)
		
		eps = Variable(torch.randn(1, z_dim))
                z = z_mu + torch.exp(z_var / 2) * eps

		return decoder(z),z_mu,z_var,z

def train():
	vae=VAE()
	vae_optimizer=optim.Adam(vae.parameters(),lr=1e-3)
	reconloss=nn.MSELoss()


	for i in xrange(casenum):
		vae_optimizer.zero_grad()
		context=infodata[i]
		test=testdata[i][0]
		target=testdata[i][1]
		outputs,z_mu,z_var,z=vae(context,test)
		klloss=torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
		loss=reconloss(outputs,target)+klloss
		loss.backward()
		vae_optimizer.step()
	return



fil=pd.read_csv('./test.txt')
	
i=0
data=[]
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
	inputx=[]
	inputy=[]
		
	while checknotendround(fil.ix[i]):
                if ((j%2)==0):
                        inputy[numy]=info+readaction(fil.ix[i])
			numy+=1
		else:
			target[numy-1]=readaction(fil.ix[i])
			inputx[numx]=inputy[num-1]+target[numy-1]
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
                        inputx[numx]=inputy[num-1]+target[numy-1]
                        numx=numx+1

		if (j%2==0):
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
                        inputx[numx]=inputy[num-1]+target[numy-1]
                        numx=numx+1

                if (j%2==0):
                j=j+1
                i=i+2
	
	infodata=infodata+[inputx]
	testdata=testdata+[[inputy]+[target]]
	casenum=casenum+1
	
	if (fil.ix[i][0]==-1):
		i=i+1


train()
	
	




			
