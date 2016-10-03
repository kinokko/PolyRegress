#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

#TODO: Impelment Linera Basis Function Regression with polynormial basis functions
degree = 6
phi_train = np.ones((np.size(x_train, 0), 1))

# w = [[0 for y in range(np.size(x_train[0]))] for z in range(degree + 1)]
# w = np.zeros((degree + 1, np.size(x_train[0]), 1))
for i in range(1, degree + 1):
    phi_train = np.append(phi_train, np.power(x_train, i), 1)

w = np.dot(np.linalg.pinv(phi_train), t_train)
print(np.shape(w))

# for i in range(np.size(t_test)):
#     for j in range(degree + 1):
#         t_trained[i] = np.add(t_trained[i], np.dot(np.transpose(w[j]), np.transpose(np.power(x_test[i], j))))

# test_err = np.subtract(t_trained, t_test)   
# print(test_err)

# Produce a plot of results.
# plt.plot(train_err.keys(), train_err.values())
# plt.plot(test_err.keys, test_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
