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

def Derivative_beta(beta0, beta, x, y):
	n=X.shape[0]
	num = 
	


def Grad_Descent(beta, T, eta):



	
def main():
	X=all_data['data']
	Y=all_data['labels']	




if __name__ == "__main__":
        main()

