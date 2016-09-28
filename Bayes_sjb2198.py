from __future__ import division
from scipy.io import loadmat
import time
import numpy as np
import random
news = loadmat('news.mat')
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.sparse as sps

TestData=news['testdata'];
TestLabels=news['testlabels']
TrainData=news['data']
TrainLabels=news['labels']


pi=np.array([])
mu = np.array([])
#train
n=TrainData.shape[1]
for ii in range(20):
	ind = (TrainLabels.reshape(TrainLabels.shape[0],) ==(ii+1))
	pi=np.append(pi, np.sum(TrainData.toarray()[ind,:])/n)
	mu=np.append(mu, (1+np.sum(TrainData.toarray()[ind,:],axis=0))/(2+np.sum(ind)))
	print ii
mu=mu.reshape(20,n)

print pi.shape
print mu.shape

mylen = TestData.shape[0]
#duplicate values for pi and X
PI = np.tile(pi,(mylen,1))
print PI.shape
X  = np.tile(TestData, (20,1))
print X.shape
MU = np.tile(mu, (mylen,1,1))
print MU.shape

result = np.log(PI) + np.sum( X * np.log(MU)+ (1-X)*np.log(1-MU), axis=1) 
