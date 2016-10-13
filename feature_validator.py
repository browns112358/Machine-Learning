from __future__ import division
import numpy as np
import scipy.sparse as sps
import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec
from numpy.random import shuffle
from sklearn.naive_bayes import MultinomialNB

#Data Representations
def Ngram(Data,n):
	Vec =CV(ngram_range=(n,n), min_df=1)
	output = Vec.fit_transform(Data)
	return output
	
def TfIdf(Data):
	Vec = TFIDFVec()
	output = Vec.fit_transform(Data)
	return output

#Learning Methods
class Ave_Perceptron:
        def __init__(self, result =np.array([])):
                self.result=result
        def predict(self, X):
                #test
                output=np.array([])
 		for ii in range(X.shape[0]):
			the_x = X[ii,:].toarray().reshape(self.W.shape)
			output= np.append(output, (np.dot(self.W, the_x))>0)
		self.output=output
                return self.output
        def classify(self, inputs):
                return self.predict(inputs)
        def fit(self, X, y, **kwargs):
              	#Set 0 class to -1
		y[y==0]=-1
		#randomly shuffle (returns random indices
		s_indx=np.arange(len(y))
		shuffle(s_indx)
		#begin first pass.  Note: this weight will only be used as a seed for the second pass
		w=np.zeros([X.shape[1]])
		for ii in range(len(y)):
			the_x=X[s_indx[ii],:].toarray().reshape(w.shape)
			the_val=np.multiply(y[s_indx[ii]],np.dot(w,the_x))	
			if (the_val <= 0):
				w=w+y[s_indx[ii]]*the_x
		#begin 2nd pass
		w=w/len(y)
		shuffle(s_indx)
		for jj in range(len(y)):
			the_x=X[s_indx[ii],:].toarray().reshape(w.shape)	
			my_val=np.multiply(y[s_indx[jj]], np.dot(w, the_x))
			if (my_val <= 0):
				w= w +y[s_indx[jj]] * the_x
		w=w/jj
		self.W = w
	
        def get_params(self, deep = False):
                return {'result':self.result}
        def score(self, X, Y_true):
                Y=self.classify(X)
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
		output=np.array([])
		for ii in range(X.shape[0]):
			N=X[ii,:].toarray().reshape(self.my_mu.shape)
			val= np.sum(np.multiply(N,self.my_mu))+self.my_pi
			if (val>=0):
				output=np.append(output, 1)
			else:
				output=np.append(output,0)
		self.output=output
		return self.output

	def classify(self, inputs):
        	return self.predict(inputs)		

	def fit(self, X, y, **kwargs):
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

    	def get_params(self, deep = False):
        	return {'result':self.result}

	def score(self, X, Y_true):
		Y=self.classify(X)
		corr = np.array([])
		for gg in range(len(Y_true)):
			corr =np.append(corr, (Y[gg] == Y_true[gg]))
		self.score = (1-np.mean(corr))*100.0
		return self.score

#main code
def main():
	#Reading in Data
	train = pd.read_csv('reviews_tr.csv', sep=',', header=0, dtype={'a': np.float32, 'b': str})
	TrainLabels=train.values[:,0].astype(np.int32)
	TrainData  =train.values[:,1]
	print 'done reading training data'

	
	#Unigram
	X=Ngram(TrainData,1)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	scores=cross_val_score(NBayes(),X, TrainLabels, cv=5)
	print("Accuracy of Bayes with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	scores=cross_val_score(MultinomialNB(), X, TrainLabels, cv=5)
	scores=(1-scores)*100
	print("Accuracy of Gaussian Bayes with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


	# Bigram
	X=Ngram(TrainData,2)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Bigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	scores=cross_val_score(MultinomialNB(), X, TrainLabels, cv=5)
	scores=(1-scores)*100
        print("Accuracy of Gaussian Bayes with Bigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	
	# Trigram
	X=Ngram(TrainData,3)
	scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
	print("Accuracy of Perceptron with Trigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        scores=cross_val_score(MultinomialNB(), X, TrainLabels, cv=5)
	scores=(1-scores)*100
        print("Accuracy of Gaussian Bayes with Trigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


        #TfIdf
        X=TfIdf(TrainData)
        scores=cross_val_score(Ave_Perceptron(),X, TrainLabels, cv=5)
        print("Accuracy of Perceptron with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        scores=cross_val_score(NBayes(),X, TrainLabels, cv=5)
        print("Accuracy of Bayes with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        scores=cross_val_score(MultinomialNB(), X, TrainLabels, cv=5)
	scores=(1-scores)*100
        print("Accuracy of Gaussian Bayes with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

	#load Test Data
	test = pd.read_csv('reviews_te.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
	TestLabels=test.values[:,0].astype(np.int32)
	TestData  =test.values[:,1]
		
	#Code to look at test error rates
	Xtotal=Ngram(np.hstack((TestData, TrainData)),3)
	Xte=Xtotal[0:len(TestData),:]
	Xtr=Xtotal[len(TestData):(len(TestData)+len(TrainData)), :]
	MBayes = MultinomialNB()
	MBayes.fit( Xtr, TrainLabels)
	Final_score=MBayes.score( Xte, TestLabels)	
	print("Test error of Multinomial Bayes with TriGram on Test data:%0.3f " % ((1-Final_score)*100.0))
	Train_score=MBayes.score(Xtr,TrainLabels)
	print("Train error of Multinomial Bayes with TriGram on Test data:%0.3f " % ((1-Train_score)*100.0))


if __name__ == "__main__":
	main()

