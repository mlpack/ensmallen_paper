# Implement a logistic regression training on the MNIST dataset.
#
# This is based very heavily on an example TensorFlow notebook:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb
#
# However, we have modified this implementation be only two-class, so it is
# comparable with our bandicoot example.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import functools
import sys

if len(sys.argv) != 5:
  print("args: <trainFile> <trainLabelsFile> <testFile> <testLabelsFile>")
  exit(1)

trainFile = sys.argv[1]
trainLabelsFile = sys.argv[2]
testFile = sys.argv[3]
testLabelsFile = sys.argv[4]

trainData = np.genfromtxt(trainFile, delimiter=',', dtype=np.float64)
trainLabels = np.genfromtxt(trainLabelsFile, delimiter=',', dtype=np.long)
testData = np.genfromtxt(testFile, delimiter=',', dtype=np.float64)
testLabels = np.genfromtxt(testLabelsFile, delimiter=',', dtype=np.long)

# Logistic regression objective.
def logistic_regression(theta, x, y):
    # Apply softmax to normalize the logits to a probability distribution.
    objReg = 0.5 / 2.0 * tf.tensordot(theta[1:], theta[1:], 1)
    sigmoids = 1.0 / (1.0 + tf.exp(tf.math.minimum(50.0, tf.matmul(x,
        tf.expand_dims(theta[1:], axis=-1)) + theta[0])))
    innerSecondTerm = 1.0 - np.expand_dims(y, axis=-1) + \
        tf.math.multiply(sigmoids, np.expand_dims(2.0 * y - 1.0, axis=-1))
    return objReg - tf.reduce_sum(tf.math.log(innerSecondTerm))

def logistic_regression_pred(x, theta):
    predictions = tf.floor(1.0 / (1.0 + tf.exp(tf.math.minimum(50.0,
        tf.matmul(x, tf.expand_dims(theta[1:], axis=-1)) + theta[0]))) + 0.5)
    return predictions

def lr_train_obj(theta):
    return logistic_regression(theta, trainData, trainLabels)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e.
    # argmax).
    correct_prediction = tf.equal(tf.cast(y_pred, tf.float64),
        tf.expand_dims(tf.cast(y_true, tf.float64), axis=-1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# Run training for the given number of steps.
theta_initial = np.zeros(np.shape(trainData)[1] + 1)

def make_val_and_grad_fn(value_fn):
    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)
    return val_and_grad

start = time.perf_counter()
results = tfp.optimizer.lbfgs_minimize(
    make_val_and_grad_fn(lr_train_obj),
    initial_position=theta_initial,
    num_correction_pairs=10,
    tolerance=1e-8,
    max_iterations=10)
end = time.perf_counter()
print(f"Training took {end - start:0.6f} seconds.")

print('L-BFGS Results')
print('Converged:', results.converged)
print('Number of iterations:', results.num_iterations)

# Test model on training set.
pred = logistic_regression_pred(trainData, results.position)
print("Train accuracy: %f." % accuracy(pred, trainLabels))

# Test model on validation set.
pred = logistic_regression_pred(testData, results.position)
print("Test accuracy: %f." % accuracy(pred, testLabels))
