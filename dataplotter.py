from __future__ import division
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
all_data = loadmat('hw4data.mat')
from mpl_toolkits.mplot3d import Axes3D

def main():

	A=np.array([[2,-2,3], [-100, 100, -100], [3, -2, 2]])
#	A=np.random.rand(3,3)

	X=all_data['data']
	Y=all_data['labels']
	
	X=np.dot(X,A)
	Y=Y.reshape([Y.shape[0],])

	X1=(X[Y==1, 0])
	X0=(X[Y==0, 0])
	Y1=(X[Y==1, 1])
	Y0=(X[Y==0, 1])
	Z1=(X[Y==1, 2])
	Z0=(X[Y==0, 2])

	#3d plot	
	fig=plt.figure()
	ax= fig.add_subplot(111, projection='3d')
	ax.plot(X0, Y0, Z0, 'r.', label = '0')
	ax.plot(X1, Y1, Z1, 'b.', label = '1')
	ax.legend()
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show()
	
	#xy plot
	plt.plot(X0,Y0, 'r.', label = '0')
	plt.plot(X1, Y1, 'b.', label ='1')
	plt.legend()
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

	#yz plot
	plt.plot(Y0, Z0, 'r.', label='0')
	plt.plot(Y1, Z1, 'b.', label='1')
	plt.legend()
	plt.xlabel('Y')
	plt.ylabel('Z')
	plt.show()

	#some stats
	print "Mean of class 0 "
	print X0.mean()
	print Y0.mean()
	print Z0.mean()

	print "Mean of Class 1"
	print X1.mean()
	print Y1.mean()
	print Z1.mean()

if __name__ == "__main__":
        main()

