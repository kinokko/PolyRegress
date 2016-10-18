#!/usr/bin/env python

import numpy as np

def GetDesignMatrix(degree, inputMatrix):
    numRows = np.size(inputMatrix, 0)
    outMatrix = np.ones((numRows, 1))
    for i in range(1, degree + 1):
        outMatrix = np.append(outMatrix, np.power(inputMatrix, i), 1)
    return outMatrix
def GetW(phi, t):
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