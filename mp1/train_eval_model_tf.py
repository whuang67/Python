"""
Train model and eval model helpers for tensorflow implementation.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=100, batch_size=16,
                num_steps=20000, shuffle=True):
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

    
    for i in range(num_steps//batch_epoch_num):
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
    feed_dict = {model.x_placeholder: image_batch,
                 model.y_placeholder: label_batch,
                 model.learning_rate_placeholder: learning_rate}
    
    f = model.forward(model.x_placeholder)
    model.loss_tensor = model.loss(f, model.y_placeholder)
    model.update_opt_tensor = model.update_op(model.loss_tensor,
                                 model.learning_rate_placeholder)
    
    loss_, _ = model.session.run([model.loss_tensor,
                                  model.update_opt_tensor],
                          feed_dict = feed_dict)
    print(loss_)
    
    '''
    f = model.forward(image_batch)
    loss_val = model.loss(f,label_batch)
    model.update_op(loss_val, learning_rate)
    '''

def eval_model(data, model):
    import tensorflow as tf
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    feed_dict = {model.x_placeholder: data["image"],
                 model.y_placeholder: data["label"]}
    y_pred_raw = model.forward(model.x_placeholder)
    
    y_pred = model.predict(y_pred_raw)
    acc_ = tf.equal(tf.transpose(y_pred), model.y_placeholder)
    # print(y_pred, data["label"])
    loss_ = model.loss(y_pred_raw, model.y_placeholder)
    # acc = (model.predict(y_pred) == data["label"]).mean()
    loss, acc__ = model.session.run([loss_, acc_],
                             feed_dict = feed_dict)
    
    acc = np.mean(acc__)
    return loss, acc
