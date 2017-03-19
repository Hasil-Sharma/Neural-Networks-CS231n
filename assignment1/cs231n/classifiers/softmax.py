import numpy as np
from random import shuffle
from IPython.core.debugger import Tracer
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      scores = np.exp(scores)
      correct_class_score = scores[y[i]]
      all_class_score = np.sum(scores)
      prob = correct_class_score/all_class_score
      local_loss = -np.log(prob)
      for j in xrange(num_classes):
          p = scores[j]/all_class_score
          dW[:, j] += (p - (j==y[i]))*X[i]
      loss += local_loss
  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  num_classes = W.shape[1]
  dW = np.zeros_like(W)
  # Tracer()()
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # each row gives score for the classes j(s) for the image i
  log_c = np.max(scores, axis = 1)
  scores -= log_c[:, None]
  exp_scores = np.exp(scores)
  exp_sum = np.sum(exp_scores, axis = 1)

  class_prob_per_image = exp_scores/exp_sum[:, None]
  yi_prob_per_image = class_prob_per_image[ np.arange(num_train), y ]

  loss += -np.sum(np.log(yi_prob_per_image))

  class_prob_per_image[ np.arange(num_train), y ] -= 1.0

  dW = X.T.dot(class_prob_per_image)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
