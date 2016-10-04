#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr

#This is the only place different
def GetDesignMatrix(degree, inputMatrix):
    numRows = np.size(inputMatrix, 0)
    outMatrix = np.ones((numRows, 1))
    for i in range(1, degree + 1):
        outMatrix = np.append(outMatrix, np.power(inputMatrix, i), 1)
    return outMatrix

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# x = a1.normalize_data(x)
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
test_err = {}
train_err = {}
feature = 4
x_train = x[0:N_TRAIN,feature]
x_test = x[N_TRAIN:,feature]