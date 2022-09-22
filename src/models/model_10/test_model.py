import numpy as np
from tensorflow.keras.models import load_model


def test_model(model, data, labels):
    test_accuracy(model, data, labels)


def test_accuracy(model, data, labels):
    net = load_model(model)
    pred = net.predict(data)

    labels = np.array(labels.flatten())
    indices = pred > 0.5
    indices = indices.astype(np.int8).flatten()

    correct = (labels == indices)

    print("# of correct values: ", np.count_nonzero(correct))
    print("#: ", len(correct))
    missclassified = np.count_nonzero(correct)/len(correct)
    print("Acc: ", missclassified*100)
