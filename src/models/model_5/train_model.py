import os
import numpy as np
from config.definitions import ROOT_DIR
from data.make_dataset import shuffle_dataset
from features.build_features import generate_class_weights
from features.CustomDataGenerator_5s import CustomDataGenerator
from helper.create_folder import create_folder
from data.partition_dataset import get_dataset_partitions
import pandas as pd
from models.model_5.build_model import build_func_cnn

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import tensorflow as tf
from visualization.visualize import plot_loss
from models.model_5.test_model import test_model


def load_data():
    # Load colorplot datasets.
    data_path = os.path.join(
        ROOT_DIR, '../data/processed/data_5/data.npz')
    label_path = os.path.join(
        ROOT_DIR, '../data/processed/data_5/Labels_v2.xlsx')

    data = np.load(
        data_path)['a'].reshape(10356, 200, 50, 1)
    print(data.shape)

    # Creating the labels.
    labels = pd.read_excel(label_path,
                           header=None).values
    print(np.unique(labels))
    return data, labels


def train(version, cross_val=True):
    """ Set up for training of the model

    Args:
        version (string): version
        cross_val (bool, optional): Preform cross validation. Defaults to True.
    """
    data, labels = load_data()

    data, labels = shuffle_dataset(data, labels, len(labels))

    if cross_val:
        kfold = KFold(n_splits=5, shuffle=True)

        fold = 1

        for train, test in kfold.split(data, labels):
            model_path = create_folder(
                '../models/model_5/v' + version + '/')
            file_name = "version_"+version+"_10sec_fold_" + \
                str(fold) + "_best_model.h5"

            class_weights = generate_class_weights(
                labels[train][:, 0])
            print(class_weights)

            train_model(data[train], labels[train], data[test],
                        labels[test], class_weights, model_path, file_name, version)

            fold = fold + 1
    else:
        train_features, train_labels, val_features, val_labels, test_features, test_labels = get_dataset_partitions(
            data, labels)

        class_weights = generate_class_weights(train_labels[:, 0])

        model_path = create_folder(
            '../models/model_5/v' + version + '/')
        file_name = "version_"+version+"_5sec_best_model.h5"

        train_model(train_features, train_labels,
                    val_features, val_labels, class_weights, model_path, file_name, version)

        model_path = "../models/model_5/vfinal/version_final_5sec_best_model.h5"
        test_model(model_path, test_features, test_labels)


def train_model(train_ds, train_labels, val_ds, val_labels, class_weights, model_path, file_name, version):
    """ Train NN model

    Args:
        train_ds (list):  training dataset
        train_labels (list): training labels
        val_ds (list): validation dataset
        val_labels (list): validation labels
        class_weights (list): class weights
        model_path (string): model path
        file_name (string): file name for trained model
        version (string): version
    """
    net = build_func_cnn(version)
    train_labels = np.asarray(train_labels).astype('float32').reshape((-1, 1))

    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-06,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    file_path = os.path.join(
        model_path, file_name)

    es = [EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0.01), tf.keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    )]

    # net = build_cnn(version)
    net.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt, metrics="accuracy")

    train_loader = CustomDataGenerator(
        train_ds, train_labels, 64, (10356, 200, 50, 1))
    valid_loader = CustomDataGenerator(
        val_ds, val_labels, 64, (10356, 200, 50, 1))

    history = net.fit(
        train_loader,
        validation_data=valid_loader,
        verbose=1, epochs=10000,
        class_weight=class_weights, callbacks=es)

    plot_loss(history)
