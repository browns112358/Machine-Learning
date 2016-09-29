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
	pi = np.linalg.norm(TrainData.toarray()[ind,:])/n
	mu =  (1+np.sum(TrainData.toarray()[ind,:],axis=0))/(2+np.sum(ind))

	#run
	mylen = TrainData.shape[0]
	X = TrainData.toarray()
	MU=np.tile(mu, (mylen,1))
	PI = pi * np.ones([mylen])
	
 	
	result = np.append(result, np.log(PI) + np.sum( (X * np.log(MU))+ ((1-X)*np.log(1-MU)), axis=1))

result = result.reshape(20, mylen)
output = 1+np.argmax(result, axis=0)


#get Error Rates
corr = np.array([])
for gg in range(mylen):
	corr =np.append(corr,(output[gg] ==TrainLabels[gg]))
print (np.mean(corr))*100.0

#fmt = '{:<8}{:<20}{}'
#for hh in range(50):
#	print(fmt.format( output[hh], TestLabels[hh], corr[hh]))

