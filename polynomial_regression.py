#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

def GetDesignMatrix(degree, inputMatrix):
    numRows = np.size(inputMatrix, 0)
    outMatrix = np.ones((numRows, 1))
    for i in range(1, degree + 1):
        outMatrix = np.append(outMatrix, np.power(inputMatrix, i), 1)
    return outMatrix
def GetW(phi, t):
    a = np.linalg.pinv(phi)
    w = np.dot(np.linalg.pinv(phi), t)
    return w
def GetPredict(w, phi):
    predict = np.dot(phi, w)
    return predict
def GetRMSE(prediction, target):
    err = np.subtract(prediction, target)
    sqrErr = np.power(err, 2)
    mean = np.mean(sqrErr)
    rms = np.sqrt(mean)
    return rms 

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

#TODO: Impelment Linera Basis Function Regression with polynormial basis functions
train_err = {}
test_err = {}
degreeRange = range(1, 30)

for degree in degreeRange:
    phi_train = GetDesignMatrix(degree, x_train)
    w = GetW(phi_train, t_train)
    prediction_train = GetPredict(w, phi_train)
    train_err[degree] = GetRMSE(prediction_train, t_train)
    phi_test = GetDesignMatrix(degree, x_test)    
    prediction_test = GetPredict(w, phi_test)    
    test_err[degree] = GetRMSE(prediction_test, t_test)    

# Produce a plot of results.
plt.plot(test_err.keys(), test_err.values())
plt.plot(train_err.keys(), train_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()

