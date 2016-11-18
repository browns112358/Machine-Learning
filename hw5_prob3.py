from __future__ import division
from scipy.io import loadmat
import numpy as np
import random
import ols
from sklearn.linear_model import Lars
import matplotlib.pyplot as plt
from matplotlib import cm
all_data = loadmat('housing.mat')
X = all_data['data']
Y = all_data['labels']
X_te = all_data['testdata']
Y_te = all_data['testlabels']

def main():
	#OLS
	names=np.array(['CONST','CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSAT'])
	OLS = ols.ols(Y, X, 'MEDV', names)
	guess= np.inner(X_te, np.transpose(OLS.b))
	print ("OLS Mean squared Error : %f" % np.mean(np.power(Y_te-guess,2)))	

	#Sparse
	LARS = Lars(n_nonzero_coefs=4, fit_intercept=False)
	LARS.fit(X, Y)
	print 'LARS'
	print(names[LARS.coef_ != 0])
	guess=LARS.predict(X_te)
	error=np.mean(np.power(Y_te-guess,2))
	print("LARS Mean Squared Error : %f" % error)

if __name__ == "__main__":
        main()

