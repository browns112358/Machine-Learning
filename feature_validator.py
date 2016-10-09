from __future__ import division
import numpy as np
import scipy.sparse as sps
import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec
from numpy.random import shuffle

#Data Representations
def Ngram(Data,n):
	Vec =CV(ngram_range=(n,n), min_df=1)
	output = Vec.fit_transform(Data)
	return output
	

def TfIdf(Data):
	Vec = TFIDFVec()
	return Vec.fit_transform(Data)


#Learning Methods
class Ave_Perceptron:
        def __init__(self, result =np.array([])):
                self.result=result
        def predict(self, X):
                #test
                print 'predicting'
                output=np.array([])
 		for ii in range(X.shape[0]):
			output= np.append(output, (np.dot(self.W, X[ii,:]))>0)
                return self.output
        def classify(self, inputs):
                print 'Classifing'
                return self.predict(inputs)
        def fit(self, X, y, **kwargs):
                print 'fitting'
              	#Set 0 class to -1
		y[y==0]=-1
		#randomly shuffle (returns random indices
		s_indx=shuffle(np.arange(len(y)))
		#begin first pass.  Note: this weight will only be used as a seed for the second pass
		w=zeros(X.shape[1])
		for ii in range(len(y)):
			if ((y[s_indx[ii]]*np.dot(w, X[s_indx[ii],:])) <= 0):
				w=w+y[s_indx[ii]]*X[s_indx[ii],:]
		#begin 2nd pass
		w=w/len(y)
		s_inx=shuffle(np.arange(len(y)))
		for jj in range(len(y)):
			my_val=y[s_indx[jj]]*np.dot(w, X[s_indx[jj],:])
			if (my_val <= 0):
				w= w +y[s_indx[jj]] * X[s_indx[jj],:]/jj
			else:
				w = w + w/jj
		self.W = w
			

        def get_params(self, deep = False):
                return {'result':self.result}
        def score(self, X, Y_true):
                Y=self.classify(X)
                print 'scoring'
                corr = np.array([])
                for gg in range(len(Y_true)):
                        corr =np.append(corr, (Y[gg] == Y_true[gg]))
                self.score = (1-np.mean(corr))*100.0
                return self.score

class NBayes:
	def __init__(self, result =np.array([])):
		self.result=result

	def predict(self, X):
		#test
		print 'predicting'
		output=np.array([])
		for ii in range(X.shape[0]):
			N=X[ii,:].toarray()
			val= np.sum(N+self.my_mu)+self.my_pi
			if (val>0):
				output=np.append(output, 1)
			else:
				output=np.append(output,0)
		self.output=output
		return self.output

	def classify(self, inputs):
		print 'Classifing'
        	return self.predict(inputs)		

	def fit(self, X, y, **kwargs):
		print 'fitting'
		n=X.shape[1]
		pi=np.array([0,0])
		mu=np.array([np.zeros([n]), np.zeros([n])])
		for ii in range(2):
			#train
			ind = (y ==(ii))
			pi[ii] = (X[ind,:].sum())/n
			mu[ii] =  (1+X[ind,:].sum(axis=0))/(2+np.sum(ind))
		self.mu=mu
		self.pi=pi
		self.my_mu=mu[1]-mu[0]
		self.my_pi=pi[1]-pi[0]
		print 'done fitting'

    	def get_params(self, deep = False):
        	return {'result':self.result}

	def score(self, X, Y_true):
		Y=self.classify(X)
		print 'scoring'
		corr = np.array([])
		for gg in range(len(Y_true)):
			corr =np.append(corr, (Y[gg] == Y_true[gg]))
		self.score = (1-np.mean(corr))*100.0
		return self.score

#other Functions
def PerError(TeL, result):
	corr = np.array([])
	for gg in range(mylen):
		corr =np.append(corr,(output[gg] ==TrainLabels[gg]))
	return (1-np.mean(corr))*100.0

#main code
def main():
	#Reading in Data
	train = pd.read_csv('reviews_tr.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
	TrainLabels=train.values[:,0]
	TrainData  =train.values[:,1]
	print 'done reading training data'

	#Unigram
	X=Ngram(TrainData,1)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	scores=cross_val_score(NBayes(),X, TrainLabels, cv=5)
	print("Accuracy of Bayes with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        #TfIdf
        X=TfIdf(TrainData)
        scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	scores=cross_val_score(NBayes(),X, TrainLabels, cv=5)
        print("Accuracy of Bayes with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	# Bigram
	X=Ngram(TrainData,2)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Bigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	# Trigram
	X=Ngram(TrainData,3)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Trigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


	test = pd.read_csv('reviews_te.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
	TestLabels=test.values[:,0]
	TestData  =test.values[:,1]

if __name__ == "__main__":
	main()


