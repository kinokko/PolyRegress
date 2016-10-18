#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr

def GetRegularlizedW(currentLamda, phi, target):
    size = np.size(phi, 1)
    regularlized = np.add(np.multiply(currentLamda, np.identity(size)), np.dot(np.transpose(phi), phi))
    w = np.dot(np.dot(np.linalg.inv(regularlized), np.transpose(phi)), target)
    return w

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
validate_err = range(len(lambdas))

for i in range(len(lambdas)):
    current_val_err = range(fold)

    for current_fold in range(fold):
        # splite the inputs and targets into training set and validation set        
        firstBoundary = current_fold * N_TRAIN / fold
        secondBoundary = (current_fold + 1) * N_TRAIN / fold
        x_train_current = np.vstack((x_train[:firstBoundary], x_train[secondBoundary:]))
        t_train_current = np.vstack((t_train[:firstBoundary], t_train[secondBoundary:]))
        x_validate = x_train[firstBoundary:secondBoundary]
        t_validate = t_train[firstBoundary:secondBoundary]
            
        phi_train = pr.GetDesignMatrix(degree, x_train_current)
        w = GetRegularlizedW(lambdas[i], phi_train, t_train_current)
        phi_validate = pr.GetDesignMatrix(degree, x_validate)
        prediction_validate = pr.GetPredict(w, phi_validate)
        current_val_err[current_fold] = pr.GetRMSE(prediction_validate, t_validate)        
    validate_err[i] = np.mean(current_val_err)

print(validate_err)
print(np.min(validate_err))
plt.semilogx(lambdas, validate_err)
plt.ylabel('RMS')
plt.legend(['Validation error'])
plt.title('L2')
plt.xlabel('Lambda')
plt.show()