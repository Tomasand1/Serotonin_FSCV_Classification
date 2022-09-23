import numpy as np
from numpy import random
import tensorflow as tf
from matplotlib import pyplot as plt


class CustomDataGenerator(tf.keras.utils.Sequence):
    """ Custom data generator used for NN training
    """

    def __init__(self, X, y,
                 batch_size,
                 input_size=(5179, 200, 100, 1),
                 shuffle=True):

        self.X = X.copy()
        self.y = y.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.X)

    def __getitem__(self, index):
        batch_images = self.X[index *
                              self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.y[index *
                              self.batch_size: (index + 1) * self.batch_size]

        X, y = self.__augment_data(batch_images, batch_labels)

        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __augment_data(self, X, y):
        """ Augment data by combining two images with the same labels. The images are first multiplied by a random number and than added.

        Args:
            X (list): dataset
            y (list): labels

        Returns:
            lists: augmented dataset and labels
        """
        filtered_data_2 = X[np.nonzero(y == 2)[0]]
        filtered_data_1 = X[np.nonzero(y == 1)[0]]
        filtered_data_0 = X[np.nonzero(y == 0)[0]]

        augmented_excit = []
        augmented_inhib = []
        augmented_non_release = []

        # Multiply two images with the same label by a random number and add them together
        for i in range(0, len(filtered_data_1)-1):
            random_number = random.uniform(0.35, 0.65)
            random_res = 1 - random_number

            out_1 = filtered_data_1[i] * random_number + \
                filtered_data_1[i+1] * random_res

            augmented_excit.append(out_1)

        augmented_excit = np.array(augmented_excit)

        # Multiply two images with the same label by a random number and add them together
        for i in range(0, len(filtered_data_2)-1):
            random_number = random.uniform(0.35, 0.65)
            random_res = 1 - random_number

            out_2 = filtered_data_2[i] * random_number + \
                filtered_data_2[i+1] * random_res

            augmented_inhib.append(out_2)

        augmented_inhib = np.array(augmented_inhib)

        # Multiply two images with the same label by a random number and add them together
        for i in range(0, len(filtered_data_0)-1):
            random_number = random.uniform(0.35, 0.65)
            random_res = 1 - random_number
            out_0 = filtered_data_0[i] * random_number + \
                filtered_data_0[i+1] * random_res
            augmented_non_release.append(out_0)

        augmented_non_release = np.array(augmented_non_release)

        # Construct lists with labels for each list with augmented data
        augmented_labels_excit = np.reshape(
            np.ones(len(augmented_excit)), (len(augmented_excit), 1))
        augmented_labels_inhib = np.reshape(
            np.full(len(augmented_inhib), 2), (len(augmented_inhib), 1))
        augmented_labels_non_release = np.reshape(
            np.zeros(len(augmented_non_release)), (len(augmented_non_release), 1))

        # Combine original data with augmented data
        final_augmented = np.concatenate(
            (augmented_excit, augmented_inhib, augmented_non_release))

        # Combine original labels with augmented data labels
        final_augmented_labels = np.concatenate(
            (augmented_labels_excit, augmented_labels_inhib, augmented_labels_non_release))

        final = np.concatenate((X, final_augmented))
        final_labels = np.concatenate((y, final_augmented_labels))

        # Shuffle the data
        p = np.random.permutation(len(final))
        final_shuffled = final[p]
        final_shuffled_labels = final_labels[p]

        return final_shuffled, final_shuffled_labels
