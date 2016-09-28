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


result=np.array([])
#train
n=TrainData.shape[1]
for ii in range(20):
	#train
	ind = (TrainLabels.reshape(TrainLabels.shape[0],) ==(ii+1))
	pi = np.sum(TrainData.toarray()[ind,:])/n
	mu =  (1+np.sum(TrainData.toarray()[ind,:],axis=0))/(2+np.sum(ind))

	#run
	mylen = TrainData.shape[0]
	X = TrainData
#	MU=np.tile(mu, (mylen,1))
	PI = pi * np.ones([mylen]) 
	
	result = np.append(result, np.log(PI) + np.sum( X.toarray() * np.log(mu)+ (1-X.toarray())*np.log(1-mu), axis=0)) 
	print ii

print result.shape
result = result.reshape(mylen, 20)

#Get Error Rates
