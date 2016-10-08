from __future__ import division
import numpy as np
import scipy.sparse as sps
import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import WordNGramAnalyzer as WNGA
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec

#Reading in Data
train = pd.read_csv('reviews_tr.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
TrainLabels=train.values[:,0]
TrainData  =train.values[:,1]
test = pd.read_csv('reviews_te.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
TestLabels=test.values[:,0]
TestData  =test.values[:,1]


#Data Representations
def Ngram(Data,n):
	Vec= CV(WNGA(max_n=n, min_n=n))
	output = Vec.fit_transform(Data)
	return output
	

def TfIdf(Data):
	Vec = TFIDFVec()
	return Vec.fit_transform(Data)


#Learning Methods
def Ave_Perceptron(TrD, TrL, TeD):
	return TeD

class NBayes:
	def __init__(self, result =np.array([])):
		self.result=result
		
	def fit(self, X, y):
		n=X.shape[1]
		result=self.result
		for ii in range(2):
			#train
			ind = (y ==(ii))
			pi = np.linalg.norm(X[ind,:])/n
			mu =  (1+np.sum(TrD[ind,:],axis=0))/(2+np.sum(ind))

			#run
			mylen = X.shape[0]
			MU=np.tile(mu, (mylen,1))
			PI = pi * np.ones([mylen])
				
			result = np.append(result, np.log(PI) + np.sum( (X * np.log(MU))+ ((1-X)*np.log(1-MU)), axis=1))

		result=result.reshape(2, mylen)
		self.output=np.argmax(result, axis=0)
		return self.output

#def Bayes(TrD, TrL, TeD):
#	result=np.array([])
#	#train
#	n=TrD.shape[1]
#	for ii in range(2):
#		#train
#		ind = (TrL ==(ii))
#		pi = np.linalg.norm(TrD[ind,:])/n
#		mu =  (1+np.sum(TrD[ind,:],axis=0))/(2+np.sum(ind))
#
#		#run
#		mylen = TeD.shape[0]
#		X = TeD
#		MU=np.tile(mu, (mylen,1))
#		PI = pi * np.ones([mylen])
#			
#		result = np.append(result, np.log(PI) + np.sum( (X * np.log(MU))+ ((1-X)*np.log(1-MU)), axis=1))
#
#	result=result.reshape(2, mylen)
#	output=np.argmax(result, axis=0)
#	return output
	
#other Functions
def PerError(TeL, result):
	corr = np.array([])
	for gg in range(mylen):
		corr =np.append(corr,(output[gg] ==TrainLabels[gg]))
	return (1-np.mean(corr))*100.0

#main code
def main():
	X=Ngram(TrainData,1)
	score=cross_val_score(NBayes(),X, TrainLabels, cv=5)
	print score

if __name__ == "__main__":
	main()


