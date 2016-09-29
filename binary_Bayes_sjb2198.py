from __future__ import division
from scipy.io import loadmat
import time
import numpy as np
import random
news = loadmat('news.mat')
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.sparse as sps

#opening the words
with open('news.vocab') as f:
	WORDS=f.readlines()

TestData=news['testdata'];
TestLabels=news['testlabels']
TrainData=news['data']
TrainLabels=news['labels']

#Heres where it becomes binary
TestLabels=TestLabels.reshape(TestLabels.shape[0],)
TrainLabels=TrainLabels.reshape(TrainLabels.shape[0],)
TestNegInd = np.logical_or(TestLabels==1, np.logical_or(TestLabels==16,TestLabels==20))
TestPosInd = np.logical_or(TestLabels==17, np.logical_or(TestLabels==18, TestLabels==19))
TrainPosInd =np.logical_or(TrainLabels==17, np.logical_or(TrainLabels==18,TrainLabels==19))
TrainNegInd =np.logical_or(TrainLabels==1, np.logical_or(TrainLabels==16, TrainLabels==20))

#Relabel
TestLabels[TestNegInd]=0
TestLabels[TestPosInd]=1
TrainLabels[TrainNegInd]=0
TrainLabels[TrainPosInd]=1

#reduce data and labels
TestData= TestData.toarray()[np.logical_or(TestNegInd, TestPosInd),:]
TestLabels = TestLabels[np.logical_or(TestNegInd, TestPosInd)]
TrainData = TrainData.toarray()[np.logical_or(TrainPosInd, TrainNegInd),:]
TrainLabels = TrainLabels[np.logical_or(TrainPosInd, TrainNegInd)]
  


result=np.array([])
#train
n=TrainData.shape[1]
for ii in range(2):
	#train
	ind = (TrainLabels ==(ii))
	pi = np.linalg.norm(TrainData[ind,:])/n
	mu =  (1+np.sum(TrainData[ind,:],axis=0))/(2+np.sum(ind))

	#run
	mylen = TestData.shape[0]
	X = TestData
	MU=np.tile(mu, (mylen,1))
	PI = pi * np.ones([mylen])
	 	
	result = np.append(result, np.log(PI) + np.sum( (X * np.log(MU))+ ((1-X)*np.log(1-MU)), axis=1))

	#over hear we will create a list of the 20

	if (ii==0):
		mu1=mu

ak=np.argsort(mu1-mu)
#these are the negatives
print 'The Most negative are'
for uu in range(len(ak)-1, len(ak)-21,-1):
	print WORDS[ak[uu]]
#these are the 'positives'
print 'The Most Positives are'
for uu in range(21):
	print WORDS[ak[uu]]

result=result.reshape(2, mylen)
output=np.argmax(result, axis=0)
#get Error Rates
corr = np.array([])
for gg in range(mylen):
	corr =np.append(corr,(output[gg] ==TrainLabels[gg]))
print (1-np.mean(corr))*100.0

#fmt = '{:<8}{:<20}{}'
#for hh in range(50):#
#	print(fmt.format( output[hh], TestLabels[hh], corr[hh]))
