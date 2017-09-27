"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    batch_epoch_num = data['label'].shape[0] // batch_size
    
    if type(batch_epoch_num) == "float":
        batch_epoch_num += 1

    
    for i in range(num_steps):
        ## shuffle
        if shuffle == True:
            np.random.seed(1)
            idx = np.arange(data["label"].shape[0])
            np.random.shuffle(idx)
            data["image"] = data["image"][idx]
            data["label"] = data["label"][idx]
        
        subsets = np.array_split(data["image"], batch_epoch_num)
        labels = np.array_split(data["label"], batch_epoch_num)
    
        for image_batch, label_batch in zip(subsets, labels):
            update_step(image_batch, label_batch, model, learning_rate)
            
            # if (old_w == model.w).all():
            #     break
    
    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(image_batch)
    gradient = model.backward(f, label_batch)
    print(gradient[0])
    model.w = model.w - learning_rate*gradient
    print(model.w[0:4])
    

def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    
    y_pred = model.forward(data["image"])
    print(y_pred, data["label"])
    loss = model.loss(y_pred, data["label"])
    acc = (model.predict(y_pred) == data["label"]).mean()
    return loss, acc
