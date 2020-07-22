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
from keras.models import load_model
from keras.models import model_from_json
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

dataset = pd.read_csv('DNNdata.dat', sep=' ', names=headers)
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
	#Load trained model
	model = load_model('model-DNNGP.h5',custom_objects={'RBFLayer': RBFLayer})
	return model

#List to store NRMSE values
nrmse_list = []
Throughput_mean = []

#Call KerasRegressor for the model
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=64, verbose=1)

#Iterations
iterations = 50

#Counter for epoch
epoch_count = 5

#Train and test the model
for i in range(iterations):

	#k-fold cross validation
	kfold = KFold(n_splits=10)
	results = cross_val_score(estimator, Xtrain, ytrain, cv=kfold)
	#print("Standardized: %.4f (%.4f) MSE" % (-results.mean(), results.std()))

	#Train the model
	estimator.fit(Xtrain, ytrain, epochs=epoch_count, batch_size=64, verbose=1)

	#Test the model
	prediction = estimator.predict(Xtest)

	#print(prediction)
	#print("RMSE = %6.4f" % (np.sqrt(mean_squared_error(ytest, prediction))))

	#Calculate RMSE
	rmse = np.sqrt(mean_squared_error(ytest, prediction))
	
	#Calculate Normalized RMSE
	nrmse = rmse / ((max(ytest) - min(ytest)))
	#nrmse = rmse / np.std(ytest)
	nrmse_list.append(nrmse)
	print("Iteration = %d   NRMSE = %4.2f" % (i,nrmse))

	#Define noise
	sigma = np.random.uniform(0,1)
	mu = 0.0
	noise = np.random.normal(mu, sigma)
	
	#print(epoch_count)
	#Normalized throughput
	normThroughput = prediction + abs(noise)

	#Increment of epoch counter
	epoch_count = epoch_count + 5

	#Normalized mean throughput
	Throughput_mean.append(normThroughput.mean())
	#print(f"Normalized Throughput: {normThroughput}")

#Store nrmse values in a file
np.savetxt("DNN-nrmse.dat",nrmse_list, fmt="%4.2f") 







