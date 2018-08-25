import numpy as np
from random import shuffle

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
 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(X.shape[0]):
    f_scores = X[i].dot(W)
    f_scores -= np.max(f_scores)
    e_scores = np.exp(f_scores)
    p_scores = e_scores/np.sum(e_scores)
    loss += -np.log(p_scores[y[i]])
    
    for j in xrange(W.shape[1]):
        if j == y[i]:
            dW[:,j] += (p_scores[j]-1)*X[i]
        else:
            dW[:,j] += p_scores[j]*X[i]
  dW = dW/X.shape[0] + reg*W         
  loss = loss/X.shape[0] + 0.5*reg*np.sum(W*W)
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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f_scores = X.dot(W)
  f_scores -= np.max(f_scores,axis=1,keepdims=True)
  sum_e_scores = np.sum(np.exp(f_scores),axis=1,keepdims=True)
  p = np.exp(f_scores) / sum_e_scores
  
  loss = np.sum(-np.log(p[np.arange(X.shape[0]),y]))
  loss = loss/X.shape[0] + 0.5*reg*np.sum(W*W)
  
  ind = np.zeros_like(np.exp(f_scores) / sum_e_scores)
  ind[np.arange(X.shape[0]), y] = 1
  dW = X.T.dot(np.exp(f_scores) / sum_e_scores - ind)
  dW = dW/X.shape[0] + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

