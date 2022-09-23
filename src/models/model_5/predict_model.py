from glob import glob
from operator import concat
import os
import numpy as np
import csv
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing

import re


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

    # Read data
    for file in txt_files:
        matrix = open(directory+"/"+file).read()
        matrix = np.array([item.split()
                          for item in matrix.split('\n')[:-1]]).astype('float32')
        matrix = normalize_data(matrix)
        CVs.append(matrix)
    CVs = np.array(CVs)

    print(CVs.shape)
    np.savez_compressed(
        '../data/external/comparison_dataset/5s/data.npz', a=np.array(CVs))

    # Load model and predict the data
    net = load_model(model)
    pred = net.predict(CVs)

    # Compare true labels to predicted ones
    print(pred.shape)
    print(pred)
    indices = np.argmax(pred, axis=1)
    print(indices)
    file_names = np.asarray(np.char.split(
        np.array([np.array(x[:-4]) for x in txt_files]), sep='-').tolist())
    print(file_names.shape)
    concat_list = np.append(file_names, np.append(
        pred, np.array(indices).reshape(744, 1), axis=1), axis=1)
    print(concat_list.shape)
    write_to_csv(concat_list)


def write_to_csv(data):
    """Write predicted data to csv

    Args:
        data (list): data to be stored
    """
    np.savetxt('../reports/results/model_5/comparison_data_5s.csv',
               np.char.strip(data), delimiter=",", fmt="%s")


def normalize_data(data):
    """Normalize data

    Args:
        data (list): data to be normalized

    Returns:
        list: normalized data
    """
    normalized = (data - data.mean())/(data.std())
    return normalized
