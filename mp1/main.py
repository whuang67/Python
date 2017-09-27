"""Main function for train, eval, and test.
"""
import os
os.chdir("C:/mp1")

# from __future__ import print_function
# from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.support_vector_machine import SupportVectorMachine

from train_eval_model import train_model, eval_model
from utils.io_tools import read_dataset, write_dataset
from utils.data_tools import preprocess_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_steps', 10000, 'Number of update steps to run.')
flags.DEFINE_string('feature_type', 'default', 'Feature type, supports ['']')
flags.DEFINE_string('model_type', 'linear', 'Feature type, supports ['']')


def main(_):
    """High level pipeline.
    This script performs the trainsing, evaling and testing state of the model.
    """
    learning_rate = FLAGS.learning_rate
    feature_type = FLAGS.feature_type
    model_type = FLAGS.model_type
    num_steps = FLAGS.num_steps

    # Load dataset.
    data = read_dataset('data/assignment1_data/val_lab.txt',
                        'data/assignment1_data/image_data')

    # Data Processing.
    data = preprocess_data(data, feature_type)
    print(len(data["image"][0]))
    # data["image"] = data["image"][0:100]
    # data["label"] = data["label"][0:100]
    
    # Initialize model.
    ndim = data['image'].shape[1]
    if model_type == 'linear':
        model = LinearRegression(ndim, 'ones')
    elif model_type == 'logistic':
        model = LogisticRegression(ndim, 'zeros')
    elif model_type == 'svm':
        model = SupportVectorMachine(ndim, 'zeros')

    # Train Model.
    model = train_model(data, model, learning_rate, num_steps=10000)

    # Eval Model.
    data_test = read_dataset('data/assignment1_data/test_lab.txt',
                             'data/assignment1_data/image_data')
    data_test = preprocess_data(data_test, feature_type)
    loss, acc = eval_model(data, model)

    # Test Model.
    data_test = read_dataset('data/assignment1_data/test_lab.txt',
                             'data/assignment1_data/image_data')
    data_test = preprocess_data(data_test, feature_type)

    # Generate Kaggle output.
    f = model.forward(data_test["image"])
    data_test["label"] = model.predict(f)
    write_dataset("kaggle.txt", data_test)

if __name__ == '__main__':
    tf.app.run()
