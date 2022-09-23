# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import os
import tensorflow as tf


def normalise(data_path):
    """Normalize data set and save it to txt files

    Args:
        data_path (string): path containing data
    """

    directory = data_path
    all_files = os.listdir(directory)
    txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))
    CVs = []
    for file in txt_files:
        matrix = open(directory+"/"+file).read()
        matrix = np.array([item.split()
                          for item in matrix.split('\n')[:-1]]).astype('float32')
        matrix = (matrix - matrix.mean())/(matrix.std())
        CVs.append(matrix)
    CVs = np.array(CVs)

    counter = 1
    for c in CVs:
        np.savetxt(str(counter)+'.txt', c, delimiter="\t")
        counter = counter + 1


def read_txt(filepath, output_filepath):
    """Read txt files and save as np data

    Args:
        filepath (string): path containing txt files
        output_filepath (string): path for output data
    """
    directory = filepath
    all_files = os.listdir(directory)
    txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))
    CVs = []
    for file in txt_files:
        matrix = open(directory+"/"+file).read()
        matrix = np.array([item.split()
                          for item in matrix.split('\n')[:-1]]).astype('float32')
        # Normalize the size of the matrix to 200x600.
        matrix = scipy.ndimage.zoom(matrix, [200/matrix.shape[0], 1])
        for i in range(0, matrix.shape[0]):
            matrix[i] = butter_highpass_filter(matrix[i], 0.05, 10, order=5)
        CVs.append(matrix)

    CVs = np.array(CVs)
    CVs_linear = CVs[0]
    for i in range(1, len(CVs)):
        CVs_linear = np.concatenate((CVs_linear, CVs[i]), axis=1)

    CVs_10s = np.hsplit(CVs_linear, CVs_linear.shape[1]/100)
    np.savez_compressed(output_filepath, a=np.array(CVs_10s))


def make_dataset(data, labels):
    """Combine data and labels into one structure

    Args:
        data (list): dataset
        labels (list): labels

    Returns:
        array: combined dataset
    """
    dataset = [data, labels]

    return dataset


def shuffle_dataset(data, labels, length):
    """Shuffle the dataset

    Args:
        data (list): dataset
        labels (list): labels
        length (number): length of data

    Returns:
        list, list: shuffled data
    """
    shuffler = np.random.permutation(length)
    X = data[shuffler]
    y = labels[shuffler]

    return X, y
