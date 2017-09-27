"""
Implements support vector machine.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
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
        slack = 1 - np.dot(y, self.predict(f))
        derivative = -np.matmul(np.transpose(self.x), y)
        # gradient = np.zeros(self.x.shape[1])

        gradient = np.where(slack < 0, np.zeros(self.x.shape[1]), derivative)
        # gradient = (1/len(y))*np.matmul(np.transpose(self.x), y-self.predict(f))

        return gradient

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average hinge loss.
        """
        slack = 1 - np.dot(y, f)
        loss = np.where(slack > 0, slack, 0)
        return loss

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        
        pred = np.where(f< 0, -1, 1)
        return pred
