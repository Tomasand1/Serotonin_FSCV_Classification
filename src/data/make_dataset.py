# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import scipy.ndimage
import os
from scipy.signal import butter, filtfilt


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def normalise(data_path):
    directory = data_path
    # imagine you're one directory above test dir
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
    directory = filepath
    # imagine you're one directory above test dir
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
