from __future__ import division
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
all_data = loadmat('hw5data.mat')

def Objective(beta0, beta, x, y):
	n=x.shape[0]
	main= beta0 + np.inner(beta, x)
	first_term= np.sum(np.log(1+ np.exp(main)))
	sec_term=np.sum(-1*y*main)
#	print str(sec_term) + " + "+str(first_term)+"main= "+ str(main.mean())
	output=(first_term+sec_term)/n
	return output

def Derv_beta(beta0, beta, x, y):
	n=x.shape[0]
	den = 1+ np.exp(beta0 + np.inner(beta, x) )
	num = np.exp(beta0 + np.inner(beta, x) )
	divide = num/den
	first = (x * np.tile(divide, (x.shape[1],1)).T).sum(axis=0)
	second = np.dot(y.T , x)
	output=  1/n* (first - second)
	return output
	
def Derv_beta0(beta0, beta, x, y):
	n=x.shape[0]
	den = 1+ np.exp(beta0 + np.inner(beta, x) )
	num = np.exp(beta0 + np.inner(beta, x) )
	first = (num/den).sum()
	output= (1/n) * (first - y.sum())
	return output

def Prob(beta0, beta, x):
	val = 1 + np.exp(-beta0 - np.inner(beta, x))
	return 1/val

def Psubi(y, i):
	my_sum=0
	for k in range(128):
		my_sum=my_sum+y[1024*k+i]
	return my_sum/128

def MAE(beta0, beta, x, y):
	my_sum=0
	r=np.arange(x.shape[0])
	for i in range(1024):
		ind  = np.mod((r-i), 1024) ==0
		val = np.abs(Prob(beta0, beta, x[ind,:] ) - Psubi(y,i))
		my_sum=my_sum+val
	return my_sum/1024
	
def Line_search(grad, beta0, beta, x,y, switch):
	#initalize eta and a factor to multiply eta by
	el=0
	eh=20
	O=Objective(beta0, beta, x, y)
	for kk in range(100):
		eta =(el+eh)/2
		if switch:
			O_new=Objective(beta0, beta-eta*grad, x, y)
		else:
			O_new=Objective(beta0-eta*grad, beta,x, y)
		fp=O - O_new #positive indicates an improvement!
		if (fp >0):
			ETA= (eta+el)/2
			if switch:
				O_new_new=Objective(beta0, beta - ETA*grad, x,y)
			else:
				O_new_new=Objective(beta0-ETA*grad, beta, x ,y)
			test_point =O_new-O_new_new
			if (test_point >0):
				eh=eta
			else:
				el=eta
			O=O_new
		else:
			eh=eta
	return eta

def holdout(beta0, beta, x, y):
	guess= np.sign(np.inner(beta, x)+beta0)  #mx+b
	guess[guess==-1]=0
	output=1-np.mean(guess==y)
	return output			  
			

def Grad_Descent(beta0, beta, T,  x, y):
	for ii in range(T):
		d_beta0= Derv_beta0(beta0, beta, x, y)
		d_beta = Derv_beta(beta0,  beta, x, y)
		eta= Line_search(d_beta, beta0, beta, x,y, True)
		eta0=Line_search(d_beta0,beta0, beta, x,y, False)
		beta= beta - eta *d_beta
		beta0=beta0- eta0*d_beta0
	return beta0 , beta
	
def main():
	X=all_data['data'].astype(np.float32)
	Y=all_data['labels']
	Y=Y.reshape([Y.shape[0],]).astype(np.float32)

        Xte=all_data['testdata'].astype(np.float32)
        Yte=all_data['testlabels']
        Yte=Yte.reshape([Yte.shape[0],]).astype(np.float32)

	#initialize 
	beta0=np.array([0])
	beta =np.array([0,0,0])
	T=1
	error=np.array([])
	iteration=np.array([])
	counter=1
	my_error=Objective(beta0, beta, X, Y)
	bound=0.50317
	error = np.append(error, my_error)
	iteration = np.append(iteration, counter)
	print my_error	
	while my_error>bound:
		beta0, beta =Grad_Descent(beta0, beta, T, X, Y)
		my_error= Objective(beta0, beta, X, Y)
		counter=counter+1

		hout=holdout(beta0, beta, X, Y)
		print str(my_error)+ "        " + str(100.0*hout) + "%"
		error=np.append(error, my_error)	
		iteration=np.append(iteration, T*counter)
	
	hout=holdout(beta0, beta, X, Y)	     
	TITLE='Iterations: '+str(np.max(iteration)) + " Holdout Error: " +str(hout)
	print TITLE
	#print a plot of the error
	plt.plot(iteration, np.abs(error), 'r--')
	plt.plot(iteration, bound * np.ones(iteration.shape))
	plt.xlabel('iteration of Gradient Descent')
	plt.ylabel('absolute error')
	plt.title(TITLE)
	plt.show()

	mae=100*MAE(beta0, beta, Xte,Yte)
	print('MAE : %0.3f + %0.3f' % (mae.mean(), mae.std()))

if __name__ == "__main__":
        main()

