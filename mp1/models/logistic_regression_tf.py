"""Logistic regression model implemented in TensorFlow.
"""
import os
os.chdir("C:/mp1")
'''
from __future__ import print_function
from __future__ import absolute_import
'''
import numpy as np
import tensorflow as tf

from models.linear_model_tf import LinearModelTf


class LogisticRegressionTf(LinearModelTf):
    def loss(self, f, y):
        """The average loss across batch examples.
        Computes the average log loss.

        Args:
            f: Tensor containing the output of the forward operation.
            y(tf.placeholder): Tensor containing the ground truth label.
        Returns:
            (1): Returns the loss function tensor.
        """
        
        loss_tensor = tf.reduce_mean(tf.log(
                tf.add(1.0, tf.exp(-tf.multiply(f, y)))))
        
        return loss_tensor

    def predict(self, f):
        """Converts score into predictions in {-1, 1}
        Args:
            f: Tensor containing theoutput of the forward operation.
        Returns:
            (1): Converted predictions, tensor of the same dimension as f.
        """
        sigmoid = tf.nn.sigmoid(f)
        comparison = sigmoid - 0.5
        return tf.sign(comparison)
