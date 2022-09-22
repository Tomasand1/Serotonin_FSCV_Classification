from sklearn.model_selection import train_test_split
import numpy as np


def get_dataset_partitions(data, labels, val_split=0.2, test_split=0.1):
    """Split data into training, validation and test sets

    Args:
        data (list): data
        labels (list): labels
        val_split (float, optional): fraction of data to use for validation set. Defaults to 0.2.
        test_split (float, optional): fraction of data to use for test set. Defaults to 0.1.

    Returns:
        list: Split data sets
    """

    train_ds, test_ds, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.1)
    train_ds, val_ds, train_labels, val_labels = train_test_split(
        train_ds, train_labels, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    train_features = np.array(train_ds)
    val_features = np.array(val_ds)
    test_features = np.array(test_ds)

    print(train_features.shape)
    print(train_labels.shape)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels
