from __future__ import division
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
all_data = loadmat('hw4data.mat')

def Objective(beta0, beta, x, y):
	n=x.shape[0]
	main= beta0 + np.inner(beta, x)
	first_term= np.sum(np.log(1+ np.exp(main))-y*main)
	output=first_term/n
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
	X=all_data['data']
	Y=all_data['labels']
	Y=Y.reshape([Y.shape[0],])
	#Linear transformation of x
#	A = np.array([[2, -2, 3],[-100, 100, -100],[3, -2, 2]])
	A = np.array([[1, 0,0],[0,1,0],[0,0,1]])
	X=np.dot(X,A)
        #partition X
        X_train=X[0:.8*X.shape[0],:]
        X_test =X[.8*X.shape[0]:X.shape[0],:]
	Y_train=Y[0:.8*X.shape[0],]
	Y_test =Y[.8*X.shape[0]:X.shape[0],]
	#initialize 
	beta0=np.array([0])
	beta =np.array([0,0,0])
	error=np.array([])
	iteration=np.array([])
	counter=1
	my_error=Objective(beta0, beta, X, Y)
#	bound=0.65064*1.01
	bound=0.71571*1.01
	error = np.append(error, my_error)
	iteration = np.append(iteration, counter)
	
	while my_error>bound:
		beta0, beta =Grad_Descent(beta0, beta, 2**counter, X_train, Y_train)
		my_error= Objective(beta0, beta, X_test, Y_test)
		counter=counter+1

		print my_error
		error=np.append(error, my_error)	
		iteration=np.append(iteration, 2**counter)	     

	#print a plot of the error
	plt.plot(iteration, np.abs(error), 'r--')
	plt.plot(iteration, bound * np.ones(iteration.shape))
	plt.xlabel('iteration of Gradient Descent')
	plt.ylabel('absolute error')
	plt.title('Iterations: '+str(counter))
	plt.show()

if __name__ == "__main__":
        main()

