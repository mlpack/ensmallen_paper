# Implement a logistic regression training on the MNIST dataset.
#
# This is based very heavily on an example TensorFlow notebook:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb
#
# However, we have modified this implementation be only two-class, so it is
# comparable with our bandicoot example.

from __future__ import absolute_import, division, print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np
import time
import sys

# Training parameters.
lambda_ = 1e-5

if len(sys.argv) != 3:
  print("args: <dim> <points>")
  exit(1)

dim = int(sys.argv[1])
points = int(sys.argv[2])

x = np.random.rand(points, dim)
y = np.random.rand(points)

def linear_regression_loss(theta):
  theta = tf.squeeze(theta)
  pred = tf.matmul(x, tf.expand_dims(theta, axis=-1))
  mse_loss = tf.reduce_sum(tf.cast(
      tf.losses.mean_squared_error(y_true=y, y_pred=tf.transpose(pred)),
      tf.float64))
  l2_penalty = lambda_ * tf.reduce_sum(tf.square(theta))
  return l2_penalty + mse_loss

def linear_regression_loss_and_gradient(theta):
  return tfp.math.value_and_gradient(linear_regression_loss, theta)

# Run training for the given number of steps.
theta_initial = np.random.rand(dim)

start = time.perf_counter()
optim_results = tfp.optimizer.lbfgs_minimize(
    linear_regression_loss_and_gradient,
    initial_position=theta_initial,
    num_correction_pairs=10,
    tolerance=1e-8,
    max_iterations=10)
end = time.perf_counter()
print((end - start))
