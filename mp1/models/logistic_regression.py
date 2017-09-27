"""
Implements logistic regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LogisticRegression(LinearModel):
    """
    """
    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        n = -np.exp(-np.dot(y,self.predict(f)))*np.matmul(np.transpose(self.x), y)
        d = 1 + np.exp(-np.dot(y, self.predict(f)))
        
        Gradient = n/d
        # Gradient = np.matmul(np.transpose(self.x),y - self.predict(f))
        return Gradient

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
        f(numpy.ndarray): Output of forward operation, dimension (N,).
        y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
        (float): average log loss.
        """
        log_loss = np.mean(np.log(1 + np.exp(-y*f)))
        return log_loss

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        sigmoid = 1 / (1 + np.exp(-f))
        pred = np.where(sigmoid <= 0.5, -1, 1)
        return pred