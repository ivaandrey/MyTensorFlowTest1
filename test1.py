
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

tf.executing_eagerly()        # => True

tfe = tf.contrib.eager

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 10
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + 0.1*noise

def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error
def loss(error):
  return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
      error = prediction(training_inputs, weights, biases) - training_outputs
      loss_value = loss(error)
      return tape.gradient(loss_value, [weights, biases])

train_steps = 100
learning_rate = 0.1
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)


# initial loss calculation
error = prediction(training_inputs, W, B) - training_outputs
loss_value = loss(error)
print("Initial loss: {:.3f}".format(loss_value))

for i in range(train_steps):
  print("W value= {:.4f}, B value= {:.4f} ".format(W.numpy(),B.numpy()))
  error = prediction(training_inputs, W, B) - training_outputs
  dW, dB = grad( W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if abs(dW)<0.0001 and abs(dB)<0.0001:
      print("Weights and B values don''t change. Break the loop in step {:03d}".format(i))
      break
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(error)))

print("Final loss: {:.3f}".format(loss(error)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))