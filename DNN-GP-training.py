import csv
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import dense
#from tensorflow.keras.layers import Sequential
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
from sklearn.kernel_approximation import RBFSampler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from tensorflow.keras.layers import InputLayer
#from keras.layers import RandomFourierFeatures
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import GaussianNoise
#from tensorflow.keras.layers.experimental import RandomFourierFeatures 
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures
from math import sqrt
import random
from rbflayer import RBFLayer, InitCentersRandom
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
import statistics

from collections import Counter
from scipy import stats, optimize, interpolate


# add header names
headers =  ['snr', 'mimo', 'cb', 'mcs', 'gi', 'fa', 'norm_throughput']

dataset = pd.read_csv('DNNdatagen.dat', sep=' ', names=headers)
#print(dataset.head())
print(f"Shape{dataset.shape}")
print(dataset.isna().sum())
print(dataset.dtypes)

#convert imput to numpy arrays
X = dataset.drop(columns=['norm_throughput'])

y = dataset['norm_throughput'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")

def baseline_model():
	# create model
	model = Sequential()
	rbflayer = RBFLayer(30,
                        initializer=InitCentersRandom(Xtrain),
                        betas=1.0,
                        input_shape=(6,))

	model.add(rbflayer)
	model.add(Dense(50, input_dim=30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(25, kernel_initializer='normal', activation='relu'))
	model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	#model.add(GaussianNoise(0.1))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics =["mean_squared_error"])

	# Training
	model.fit(Xtrain, ytrain, batch_size = 10, epochs=10, verbose=0)
	#model.fit(X_train, y_train, batch_size = 32, epochs=20, verbose=2)
	
	scores= model.evaluate(Xtest,ytest,verbose=0)
	model.save("model-DNNGP.h5")
	return model


# Call the function
baseline_model()











