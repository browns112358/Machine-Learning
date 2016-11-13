from __future__ import division
import numpy as np
import scipy.sparse as sps
import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#Data Representations
def Ngram(Data,n):
	Vec =CV(ngram_range=(n,n), min_df=1)
	output = Vec.fit_transform(Data)
	return output
	
def TfIdf(Data):
	Vec = TFIDFVec()
	output = Vec.fit_transform(Data)
	return output

#main code
def main(n):
	print("---------- %d estimators used ----------" % n)
	#Reading in Data
	train = pd.read_csv('reviews_tr.csv', sep=',', header=0, dtype={'a': np.float32, 'b': str})
	TrainLabels=train.values[:,0].astype(np.int32)
	TrainData  =train.values[:,1]
	print 'done reading training data'

	#classifiers
	RF = RandomForestClassifier(n_estimators=n, max_depth=None, min_samples_split=2, random_state=0)
	AB = AdaBoostClassifier(n_estimators=n)	

	#Unigram
	X=Ngram(TrainData,1)
	scores=cross_val_score(RF, X, TrainLabels, cv=5)
	print("Accuracy of Random Forest with Unigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	scores=cross_val_score(AB, X, TrainLabels, cv=5)
	print("Accuracy of AdaBoost with Unigram      : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        
	#TfIdf
        X=TfIdf(TrainData)
        scores=cross_val_score(RF,X, TrainLabels, cv=5)
        print("Accuracy of Random Forest with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	scores=cross_val_score(AB, X, TrainLabels, cv=5)
        print("Accuracy of AdaBoost with TfIdf : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

	#Bigram
	X=Ngram(TrainData,2)
	scores=cross_val_score(RF,X, TrainLabels, cv=5)
        print("Accuracy of Random Forest with BiGram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        scores=cross_val_score(AB, X, TrainLabels, cv=5)
        print("Accuracy of AdaBoost with Bigram : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

	#load Test Data
	test = pd.read_csv('reviews_te.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
	TestLabels=test.values[:,0].astype(np.int32)
	TestData  =test.values[:,1]
		
	#Code to look at test error rates
	Xtotal=Ngram(np.hstack((TestData, TrainData)),3)
	Xte=Xtotal[0:len(TestData),:]
	Xtr=Xtotal[len(TestData):(len(TestData)+len(TrainData)), :]
	


if __name__ == "__main__":
	main(50)
	main(100)
	main(300)

