from __future__ import division
from scipy.io import loadmat
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.sparse as sps
import pandas as pd

train = pd.read_csv('reviews_tr.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
TrainLabels=train.values[:,0]
TrainData  =train.values[:,1]
test = pd.read_csv('reviews_te.csv', sep=',', header=0, dtype={'a': np.int32, 'b': str})
TestLabels=test.values[:,0]
TestData  =test.values[:,1]
