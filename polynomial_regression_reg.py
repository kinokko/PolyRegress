#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr

def GetRMSE(prediction, target, currentLamda, w):
    regularizer = np.dot(currentLamda, np.dot(np.transpose(w), w))
    err = np.add(np.subtract(prediction, target), regularizer)
    sqrErr = np.power(err, 2)
    mean = np.mean(sqrErr)
    rms = np.sqrt(mean)
    return rms 

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

test_err = {}
train_err = {}
lambdas = [0, .01, .1, 1, 10, 100 , 1000 , 10000]
fold = 10
degree = 2
x_size = np.size(x_train, 1)
validate_err = [0, .01, .1, 1, 10, 100 , 1000 , 10000]

for i in range(len(lambdas)):
    current_val_err = range(fold)
    for current_fold in range(fold):
        x_train_current = np.empty((x_size,))
        x_validate = np.empty((x_size,))
        t_train_current = np.empty((1,))
        t_validate = np.empty((1,))
        for j in range(current_fold * N_TRAIN / fold):
            x_train_current = np.vstack((x_train_current, x_train[j]))
            t_train_current = np.vstack((t_train_current, t_train[j]))
        for j in range(current_fold * N_TRAIN / fold, (current_fold + 1) * N_TRAIN / fold):
            x_validate = np.vstack((x_validate, x_train[j]))
            t_validate = np.vstack((t_validate, t_train[j]))
        for j in range((current_fold + 1) * N_TRAIN / fold, N_TRAIN):
            x_train_current = np.vstack((x_train_current, x_train[j]))
            t_train_current = np.vstack((t_train_current, t_train[j]))
            
        phi_train = pr.GetDesignMatrix(degree, x_train_current)
        w = pr.GetW(phi_train, t_train_current)
        phi_validate = pr.GetDesignMatrix(degree, x_validate)
        prediction_validate = pr.GetPredict(w, phi_validate)
        current_val_err[current_fold] = GetRMSE(prediction_validate, t_validate, lambdas[i], w)
    validate_err[i] = np.mean(current_val_err)

print(validate_err)
print(np.min(validate_err))
plt.semilogx(lambdas, validate_err)
plt.ylabel('RMS')
plt.legend(['Validation error'])
plt.title('Fit with polynomials, with regularization')
plt.xlabel('Lambda')
plt.show()