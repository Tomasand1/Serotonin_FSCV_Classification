# Define a function to plot the loss per epoch
from matplotlib import pyplot as plt


def plot_loss(history):
    plt.figure(1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.show()

    plt.figure(2)
    plt.plot(history.history['accuracy'], label="acc")
    plt.plot(history.history['val_accuracy'], label="val.acc")
    plt.xlabel('Epoch')
    plt.ylabel('Acc.')
    plt.legend()
    plt.grid(True)

    plt.show()
