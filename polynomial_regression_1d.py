#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# x = a1.normalize_data(x)
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
test_err = {}
train_err = {}

featureRange = range(8)
for feature in featureRange:
    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]
    phi_train = pr.GetDesignMatrix(3, x_train)
    w = pr.GetW(phi_train, t_train)
    predict_train = pr.GetPredict(w, phi_train)
    train_err[feature] = pr.GetRMSE(predict_train, t_train)
    phi_test = pr.GetDesignMatrix(3, x_test)
    predict_test = pr.GetPredict(w, phi_test)
    test_err[feature] = pr.GetRMSE(predict_test, t_test)    
    

# Produce a plot of results.
plt.bar(test_err.keys(), test_err.values(), color = "r")
plt.bar(train_err.keys(), train_err.values(), width = 0.4)
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Feature number')
plt.show()