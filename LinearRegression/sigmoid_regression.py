#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_utility as pr

#This is the only place different
def GetDesignMatrix(mu, s, inputMatrix):
    numRows = np.size(inputMatrix, 0)
    outMatrix = np.ones((numRows, 1))
    for i in range(len(mu)):
        np.divide((np.subtract(mu[i], inputMatrix)), s)
        appender = np.power((np.add(1, np.exp(np.divide((np.subtract(mu[i], inputMatrix)), s)))), -1)
        outMatrix = np.append(outMatrix, appender, 1)
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
mu = [100,10000]
s = 2000.0

feature = 11 - 8
x_train = x[0:N_TRAIN,feature]
x_test = x[N_TRAIN:,feature]
phi_train = GetDesignMatrix(mu, s, x_train)
w = pr.GetW(phi_train, t_train)
predict_train = pr.GetPredict(w, phi_train)
train_err = pr.GetRMSE(predict_train, t_train)
phi_test = GetDesignMatrix(mu, s, x_test)
predict_test = pr.GetPredict(w, phi_test)
test_err = pr.GetRMSE(predict_test, t_test)  


x_train_min = np.asscalar(np.min(x_train))
x_train_max = np.asscalar(np.max(x_train))
domain_size = 500
x_ev = np.linspace(x_train_min, x_train_max, domain_size)
x_ev_designd = GetDesignMatrix(mu, s, np.reshape(x_ev, (domain_size, 1)))
y_predict = pr.GetPredict(w, x_ev_designd)
plt.plot(x_ev , y_predict)
plt.ylabel('Mortality Plot')
plt.legend(["Mortality"])
plt.show()
plt.bar(0, train_err)
plt.bar(1, test_err, color = "r")
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Train Test')
plt.show()

