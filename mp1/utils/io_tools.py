"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """

    f = open(data_txt_file, 'r')
    f1 = f.readlines()
    images = np.zeros(shape = (len(f1), 28, 28))
    labels = np.zeros(shape = len(f1))
    
    for i, e in enumerate(f1):
        p, label = e.split('\t')
        path = os.path.join(image_data_path, p)
        img = io.imread(path)
        images[i] = img
        labels[i] = label.strip("\n")


    data = {"image": images, "label": labels}
    
    return data


def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).
    """
    with open(data_txt_file, "w") as output:
        output.write("Id,Prediction\n")
        for i in range(len(data["label"])):
            if i <= 9:
                output.write("test_0000"+str(i)+".png,"+str(data["label"][i])+"\n")
            elif i <= 99:
                output.write("test_000"+str(i)+".png,"+str(data["label"][i])+"\n")
            elif i <= 999:
                output.write("test_00"+str(i)+".png,"+str(data["label"][i])+"\n")
            elif i <= 9999:
                output.write("test_0"+str(i)+".png,"+str(data["label"][i])+"\n")

