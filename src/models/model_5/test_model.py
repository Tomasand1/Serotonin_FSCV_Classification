import numpy as np
from tensorflow.keras.models import load_model


def test_model(model, data, labels):
    test_accuracy(model, data, labels)


def test_accuracy(model, data, labels):
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
