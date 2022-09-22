# Import libraries
import os
from config.definitions import ROOT_DIR
from data.make_dataset import make_dataset, shuffle_dataset
from data.partition_dataset import get_dataset_partitions
from features.build_features import generate_class_weights_multiclass
from helper.create_folder import create_folder
from models.model_cv.test_model import test_model
from models.model_cv.build_model import build_model
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from visualization.visualize import plot_loss


def load_data():

    data_path = os.path.join(
        ROOT_DIR, '../data/processed/data_cv/data.npz')
    label_path = os.path.join(
        ROOT_DIR, '../data/processed/data_cv/Labels.xlsx')

    dataset = np.load(data_path)['a'].reshape(400, 200, 100)
    labels = pd.read_excel(label_path, header=None).values

    # Normalization of dataset.
    CVs = []
    for matrix in dataset:
        for z in range(0, 100):
            CVs.append((matrix[:, z] - matrix[:, z].mean()) /
                       (matrix[:, z].std()))
    data = np.array(CVs)

    return data, labels


def train(version, cross_val=True):
    data, labels = load_data()

    data, labels = shuffle_dataset(data, labels, len(labels))

    normalizer = preprocessing.Normalization(input_shape=[200, ])
    normalizer.adapt(data)
    normalizer(data)

    pass

    if cross_val:
        kfold = KFold(n_splits=5, shuffle=True)

        fold = 1

        for train, test in kfold.split(data, labels):
            model_path = create_folder(
                '../models/model_cv/v' + version + '/')
            file_name = "version_"+version+"_consecutive_fold_" + \
                str(fold) + "_best_model.h5"

            class_weights = generate_class_weights_multiclass(
                labels[train][:, 0])

            train_model(data[train], labels[train], data[test],
                        labels[test], class_weights, model_path, file_name)

            fold = fold + 1
    else:
        train_features, train_labels, val_features, val_labels, test_features, test_labels = get_dataset_partitions(
            data, labels)

        class_weights = generate_class_weights_multiclass(train_labels[:, 0])

        model_path = create_folder(
            '../models/model_cv/v' + version + '/')
        file_name = "version_"+version+"_consecutive_best_model.h5"

        train_model(train_features, train_labels,
                    val_features, val_labels, class_weights, model_path, file_name)

        model_path = '../models/model_cv/v' + version + \
            '/version_' + version + '_consecutive_best_model.h5'
        test_model(model_path, test_features, test_labels)


def train_model(train_ds, train_labels, val_ds, val_labels, class_weights, model_path, file_name):
    print("SSSSS")
    print(train_ds.shape)
    net = build_model()

    net.compile(loss='sparse_categorical_crossentropy',
                optimizer="Adam", metrics=['accuracy'])

    file_path = os.path.join(
        model_path, file_name)

    es = [EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0.001), tf.keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    )]

    history = net.fit(
        train_ds, train_labels,
        validation_data=[val_ds, val_labels],
        class_weight=class_weights,
        verbose=1, epochs=10000, callbacks=es)

    plot_loss(history)
