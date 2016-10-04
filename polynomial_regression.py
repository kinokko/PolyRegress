#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

#TODO: Impelment Linera Basis Function Regression with polynormial basis functions
train_err = {}
test_err = {}
degreeRange = range(1, 7)

for degree in degreeRange:
    phi_train = pr.GetDesignMatrix(degree, x_train)
    w = pr.GetW(phi_train, t_train)
    prediction_train = pr.GetPredict(w, phi_train)
    train_err[degree] = pr.GetRMSE(prediction_train, t_train)
    phi_test = pr.GetDesignMatrix(degree, x_test)    
    prediction_test = pr.GetPredict(w, phi_test)    
    test_err[degree] = pr.GetRMSE(prediction_test, t_test)    

# Produce a plot of results.
plt.plot(test_err.keys(), test_err.values())
plt.plot(train_err.keys(), train_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()

