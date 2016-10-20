from __future__ import division
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
all_data = loadmat('hw4data.mat')

def Objective(beta0, beta, x, y):
	n=X.shape[0]
	first_term= np.sum(np.log(1+ np.exp(beta0 + np.inner(beta, x))))
	sec_term=np.sum(y*(beta0+np.inner(beta,x)))
	output=1/n * (first_term - second_term)
	return output

def Derv_beta(beta0, beta, x, y):
	n=X.shape[0]
	den = 1+ np.exp(beta0 + np.inner(beta, x) )
	num = np.exp(beta0 + np.inner(beta, x)
	first = (x * np.tile(num/den, (1,3))).sum(axis=0)
	second = y.T * x
	output= (1/n) * (first - second)
	return output
	
def Derv_beta0(beta0, beta, x, y):
	n=X.shape[0]
	den = 1+ np.exp(beta0 + np.inner(beta, x) )
	num = np.exp(beta0 + np.inner(beta, x)
	first = (num/den).sum()
	output= (1/n) * (first - y.sum())
	return output
	

def Grad_Descent(beta0, beta, T, eta, x, y):
	for ii in range(T):
		beta0= beta0 - eta * Derv_beta0(beta0, beta, x,y)
		beta = beta  - eta * Derv_beta(beta0,  beta, x,y)
	w=np.array([[beta0], [beta]])
	return w
	
def main():
	X=all_data['data']
	Y=all_data['labels']
	#initialize 
	W=np.array([[0], [0,0,0]])
	
	W=Grad_Descent(W[0], W[1], 2, .1, X, Y)

	error= Objective(W[0], W[1], X, Y)
	print error
		     



if __name__ == "__main__":
        main()

