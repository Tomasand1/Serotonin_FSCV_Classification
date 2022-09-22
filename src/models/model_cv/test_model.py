import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def test_model(model, data, labels):
    test_accuracy(model, data, labels)
    test_model_plot(model)


def test_accuracy(model, data, labels):
    """Test model accuracy

    Args:
        model (any): model
        data (list): data
        labels (list): labels
    """
    net = load_model(model)
    pred = net.predict(data)

    labels = labels.flatten()
    print(labels)
    indices = np.argmax(pred, axis=1)
    print(indices)
    correct = (labels == indices)

    print("# of correct values: ", np.count_nonzero(correct))
    print("#: ", len(correct))
    missclassified = np.count_nonzero(correct)/len(correct)
    print("Acc: ", missclassified*100)


def test_model_plot(model):
    """ Test model using plots

    Args:
        model (any): model
    """
    net = load_model(model)
    test_df = np.transpose(pd.read_excel(
        '../data/raw/test_data/4412.xlsx', header=None).values)
    plt.figure()
    plt.plot(test_df)
    plt.figure()
    plt.imshow(test_df)
    print(test_df.shape)

    pred = net.predict(test_df)
    indices = np.argmax(pred, axis=1)
    print(indices)
    predictions = np.transpose(pred)
    print(predictions.shape)
    plt.figure()
    plt.plot(predictions[0], label="0")
    plt.plot(predictions[1], label="1")
    plt.plot(predictions[2], label="2")
    plt.legend()
    plt.show()
