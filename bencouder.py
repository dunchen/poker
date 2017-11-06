import torch
import torch.nn.functional as nn
import torch.nn as nn2
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable



mb_size = 64
Z_dim = 4
ZZ_dim=Z_dim*Z_dim*2
info_h_dim = 50
info_dim=100

act_h_dim=10
act_dim=20

Total_dim=120
Total_h_dim=30

c = 0
lr = 1e-3


def xinput(size):
    return

def ssselct(data,j,parts):
    return


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.randn(*size) * xavier_stddev


def sample_z(mu, log_var):
		eps = Variable(torch.randn(mb_size, Z_dim))
		return mu + torch.exp(log_var / 2) * eps

class Autoencoder(torch.nn.Module):
	def __init__(self,X_dim,h_dim,Z_dim):
		super(SimpleNN,self).__init__()
		self.Wxh=nn2.Parameter(xavier_init(size=[X_dim, h_dim]))
		self.bxh=nn2.Parameter(torch.zeros(h_dim))
		self.Whz = nn2.Parameter(xavier_init(size=[h_dim, Z_dim]))
		self.bhz = nn2.Parameter(torch.zeros(Z_dim))
		self.Wzh = nn2.Parameter(xavier_init(size=[Z_dim, h_dim]))
		self.bzh = nn2.Parameter(torch.zeros(h_dim))
		self.Whx = nn2.Parameter(xavier_init(size=[h_dim, X_dim]))
		self.bhx = nn2.Parameter(torch.zeros(X_dim))
	
	def Q(self,X):
		h = nn.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
		z = nn.relu(h @ self.Whz + self.bhz.repeat(h.size(0), 1))
		return z
	
	def P(self,z):
		h = nn.relu(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
		X = nn.relu(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
		return X
	
	def forward(self, X):
		z= self.Q(X)
		X_sample = self.P(z)
		return X_sample


class Prediction(torch.nn.Module):
	def __init__(self,wxh,bxh,whz,bhz,wzh,bzh,whx,bhx,wzz,bzz):
		super(SimpleNN2,self).__init__()
		self.Wxh= nn2.Parameter(wxh,False)
		self.bxh= nn2.Parameter(bxh,False)
		self.Whz= nn2.Parameter(whz,False)
		self.bhz= nn2.Parameter(bhz,False)
		self.Wzh= nn2.Parameter(wzh,False)
		self.bzh= nn2.Parameter(bzh,False)
		self.Whx= nn2.Parameter(whz,False)
		self.bhx= nn2.Parameter(bhx,False)
		self.Wzz= nn2.Parameter(wzz)
		self.bzz= nn2.Parameter(bzz)
        
	def forward(self, X)
		h = nn.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
                z = nn.relu(h @ self.Whz + self.bhz.repeat(h.size(0), 1))
		z2= nn.relu(z @ self.Wzz + self.bzz.repeat(z.size(0), 1))
		h2 = nn.relu(z2 @ self.Wzh + self.bzh.repeat(z2.size(0), 1))
                X_result = nn.relu(h2 @ self.Whx + self.bhx.repeat(h2.size(0), 1))
		return X_result

class SimpleVAE(torch.nn.Module):
        def __init__(self,X_dim,h_dim,Z_dim):
                super(SimpleNN,self).__init__()
                self.Wxh=nn2.Parameter(xavier_init(size=[X_dim, h_dim]))
                self.bxh=nn2.Parameter(torch.zeros(h_dim))
                self.Whz_mu = nn2.Parameter(xavier_init(size=[h_dim, Z_dim]))
                self.bhz_mu = nn2.Parameter(torch.zeros(Z_dim))
                self.Whz_var = nn2.Parameter(xavier_init(size=[h_dim, Z_dim]))
                self.bhz_var = nn2.Parameter(torch.zeros(Z_dim))
                self.Wzh = nn2.Parameter(xavier_init(size=[Z_dim, h_dim]))
                self.bzh = nn2.Parameter(torch.zeros(h_dim))
                self.Whx = nn2.Parameter(xavier_init(size=[h_dim, X_dim]))
                self.bhx = nn2.Parameter(torch.zeros(X_dim))

        def Q(self,X):
		h = nn.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
		z_mu = h @ self.Whz_mu + self.bhz_mu.repeat(h.size(0), 1)
		z_var = h @ self.Whz_var + self.bhz_var.repeat(h.size(0), 1)
		return z_mu, z_var

        def P(self,z):
                h = nn.relu(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
                X = nn.sigmoid(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
                return X

        def forward(self, X):
                z_mu, z_var = self.Q(X)
                eps = Variable(torch.randn(mb_size, Z_dim)).cuda()
                z = z_mu + torch.exp(z_var / 2) * eps
                return z_mu,z_var,z


autox=Autoencoder(info_dim,info_h_dim,Z_dim)
solver = optim.Adam(autox.parameters(),lr=lr)

for it in range(5000):
    X=xinput(mb_size).info
    solver.zero_grad()
    X_sample=autox(X)
    # Loss
    recon_loss=nn2.MSELoss()
    loss = recon_loss(X_sample, X) / mb_size
    # Backward
    loss.backward()
    # Update
    solver.step()


autoy=Autoencoder(act_dim,act_h_dim,Z_dim)
solver = optim.Adam(autox.parameters(),lr=lr)

for it in range(5000):
    X=xinput(mb_size).info
    solver.zero_grad()
    X_sample=autoy(X)
    # Loss
    recon_loss=nn2.MSELoss()
    loss = recon_loss(X_sample, X) / mb_size
    # Backward
    loss.backward()
    # Update
    solver.step()


vae=SimpleVAE(Total_dim,Total_h_dim,ZZ_dim)
solver = optim.Adam(vae.parameters(),lr=lr)

for it in range(5000)
	X=xinput(mb_size)
	z_mu,z_var,z=vae(X.whole)
	recon_loss=nn2.MSELoss()

	for j in mb_size
		wzz,bzz=sselect(z,j,2)
		tester=Prediction(autox.Wxh,autox.bxh,autox.Whz,autox.bhz,autoy.Wzh,autoy.bzh,autoy.Whx,autoy.bhx,wzz.view(Z_dim,Z_dim),bzz.view(Z_dim,Z_dim))
		testsolver=optim.Adam(tester.parameters(),lr=lr*10)
	
		testsolver.zero_grad()
		act=tester(sselect(X.info,j,1))
		testrecon_loss=nn2.MSELoss()
		testloss=testrecon_loss(act, sselect(X.act,j,1))
		testloss.backward()
		testsolver.step()
		testresult=torch.cat((testresult, tester.wzz, tester.bzz),1)
	
	vaeloss=recon_loss(testresult,z)/mb_size
	kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
	loss=vaeloss+kl_loss
	
	loss.backward()

	solver.step()


