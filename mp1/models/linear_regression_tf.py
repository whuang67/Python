"""Linear regression model implemented in TensorFlow.
"""

import os
os.chdir("C:/mp1")


# from __future__ import print_function
# from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_model_tf import LinearModelTf


class LinearRegressionTf(LinearModelTf):
    def loss(self, f, y):
        """The average loss across batch examples.
        Computes the average square error.
        Args:
            f: Tensor containing the output of the forward operation.
            y(tf.placeholder): Tensor containing the ground truth label.
        Returns:
            (1): Returns the loss function tensor.
        """
        #loss_tensor = tf.multiply(tf.reduce_mean(tf.square(f - y)),
        #                          tf.constant(.5))
        loss_tensor = tf.reduce_sum(tf.multiply(tf.square(
                tf.subtract(1.,tf.multiply(self.predict(f),y))),
        tf.constant(0.5)))
                
        return loss_tensor

        
    def predict(self, f):
        """Converts score into predictions in {-1, 1}
        Args:
            f: Tensor containing theoutput of the forward operation.
        Returns:
            (1): Converted predictions, tensor of the same dimension as f.
        """
        
        '''try:
            output = f.eval(session = self.session)
            what = np.where(output >= 0, 1, -1)
            predict_tensor = tf.convert_to_tensor(what)
            return predict_tensor
        except:
            return tf.placeholder(tf.float32,
                                  shape = [None])
        
        # predict_tensor = tf.where(f < 0, -1, 1)
        # return predict_tensor
        '''
        return tf.sign(f)