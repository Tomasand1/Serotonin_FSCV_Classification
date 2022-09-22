from glob import glob
import os
import re
import numpy as np
import csv
import scipy.ndimage
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def predict_model(model, directory):
    """ Predict new data using NN model

    Args:
        model (any): model
        directory (string): path to save the results
    """
    # imagine you're one directory above test dir
    all_files = sorted(os.listdir(directory), key=natural_keys)
    print(all_files)
    txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))
    CVs = []
    for file in txt_files:
        matrix = open(directory+"/"+file).read()
        matrix = np.array([item.split()
                          for item in matrix.split('\n')[:-1]]).astype('float32')
        matrix = (matrix - matrix.mean())/(matrix.std())
        CVs.append(np.transpose(matrix))
    CVs = np.array(CVs).reshape(372, 200, 100, 1)
    print(CVs.shape)

    net = load_model(model)

    pred = net.predict(CVs)

    indices = pred > 0.5
    indices = indices.astype(np.int8).flatten()
    print(indices)
    print(pred)
    print(pred.shape)
    write_to_csv(
        np.append(np.array(txt_files).reshape(372, 1), np.append(pred, np.array(indices).reshape(372, 1), axis=1), axis=1))


def write_to_csv(data):
    np.savetxt('../reports/results/model_10/comparison_data_10s.csv',
               data, delimiter=",", fmt="%10s")
