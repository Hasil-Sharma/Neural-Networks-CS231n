import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  #ipdb.set_trace(context=21)
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :]
        dW[:, y[i]] -= X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # W.shape  : (3073, 10)
  # X.shape  : (500, 3073)
  # dw.shape : (3073, 10)
  # y.shape  : (500,)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # (500, 10)
  # shape is (num_train, num_features_image)* (num_features_image, num_classes )
  # = (num_train, num_classes)
  correct_class_score = scores[np.arange(scores.shape[0]), y] # (500, )
  # correct_class_score is (num_train, )
  margin = scores - correct_class_score[:, None] + 1 # (500, 10)
  # margin is (num_train, num_classes) - (num_train, 1) + 1
  # = (num_train, num_classes)
  # This margin includes extra one

  margin[np.arange(scores.shape[0]), y] = 0
  margin[margin < 0] = 0
  # update score for right class to 0 to not include it in overall sum loss

  loss = np.sum(margin)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  binary = margin
  binary[binary > 0] = 1 # (500, 10)
  # if binary[i][j] = 1 for image i, j tells to which column in dW should X[i]
  # be added
  # Tracer()()
  col_row = np.sum(binary, axis = 1) # (500,)
  # col_row[i] for image i is the number of class of classes which contribute to
  # loss

  # For each image, i in binary setting binary[i][y_i] to -1 *
  # number of classes which contribute to loss
  binary[range(num_train), y] = -col_row[range(num_train)]

  # binary[i][j] where j != y_i is equal to 1 if jth class score contribute to
  # loss
  # binary[i][j] where j = y_i is equal to -1 * ( #classes which contribute to
  # loss )

  dW = np.dot(X.T, binary)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
