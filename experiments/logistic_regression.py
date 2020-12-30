"""
@file logistic_regression.py
@author Ryan Curtin

Use scipy to optimize the logistic regression objective function using L-BFGS
and automatic differentation.
"""
import numpy as np
import scipy.optimize
import random
import sys

if len(sys.argv) != 5:
  print("args: <trainFile> <trainLabelsFile> <testFile> <testLabelsFile>")
  exit(1)

trainFile = sys.argv[1]
trainLabelsFile = sys.argv[2]
testFile = sys.argv[3]
testLabelsFile = sys.argv[4]

trainData = np.genfromtxt(trainFile, delimiter=',', dtype=np.float64)
trainLabels = np.genfromtxt(trainLabelsFile, delimiter=',', dtype=np.float64)
testData = np.genfromtxt(testFile, delimiter=',', dtype=np.float64)
testLabels = np.genfromtxt(testLabelsFile, delimiter=',', dtype=np.float64)

def f(theta):
  objReg = 0.5 / 2.0 * np.dot(theta[1:], theta[1:])
  sigmoids = 1.0 / (1.0 + np.exp(np.minimum(300.0, -(theta[0] + np.matmul(trainData, theta[1:])))))
  innerSecondTerm = 1.0 - trainLabels + np.multiply(sigmoids, (2.0 * trainLabels - 1.0))
  result = np.sum(np.log(innerSecondTerm + 1e-10))
  return objReg - result

def g(theta):
  sigmoids = 1.0 / (1.0 + np.exp(-(theta[0] + np.matmul(trainData, theta[1:]))))
  G = np.zeros(len(theta))
  G[0] = -np.sum(trainLabels - sigmoids)
  G[1:] = (np.matmul((sigmoids - trainLabels), trainData) + 0.5 * theta[1:])
  return G

def compute_accuracy(theta, X, y):
  predictions = np.floor(1.0 / (1.0 + np.exp(-theta[0] - np.matmul(X,
      theta[1:]))) + 0.5)
  return np.sum(np.equal(predictions, y)) / len(y)

theta = np.zeros(np.shape(trainData)[1] + 1)

import timeit

def run():
  # Try to match mlpack configuration.
  return scipy.optimize.fmin_l_bfgs_b(f, theta, fprime=g, maxiter=10, maxls=50, epsilon=0.0, factr=0.0)

t = timeit.timeit(stmt=run, number=1)
print(t)

finalTheta, obj, d = run()
print(d)

print("Training accuracy:")
print(compute_accuracy(finalTheta, trainData, trainLabels))
print("Test accuracy:")
print(compute_accuracy(finalTheta, testData, testLabels))
