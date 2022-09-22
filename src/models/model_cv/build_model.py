from tensorflow.keras.layers import GaussianNoise, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from helper.save_network_structure import save_network_structure


def build_model():
    net = Sequential()
    net.add(GaussianNoise(0.5))
    net.add(Dense(200, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)))
    net.add(Dropout(0.6))
    net.add(Dense(100, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)))
    net.add(Dropout(0.4))
    net.add(Dense(50, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)))
    net.add(Dropout(0.4))
    net.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    net.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    net.add(Dense(3, activation='softmax'))

    net.build(input_shape=(32, 200))

    net.summary()
    save_network_structure(net, "CV")

    return net
