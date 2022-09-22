from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import keras
from helper.save_network_structure import save_network_structure
from tensorflow.keras import layers


def build_cnn(version):
    net = Sequential()
    net.add(Conv2D(activation='relu', filters=64,
            kernel_size=(5, 5), input_shape=(200, 100, 1)))
    net.add(BatchNormalization())
    net.add(Conv2D(128, (3, 3), activation='relu'))
    net.add(MaxPool2D(pool_size=(2, 2)))
    net.add(BatchNormalization())
    net.add(Dropout(0.4))
    net.add(Conv2D(256, (3, 3), strides=2, activation='relu'))
    net.add(MaxPool2D(pool_size=(2, 2)))
    net.add(BatchNormalization())
    net.add(Dropout(0.4))
    net.add(Conv2D(512, (3, 3), activation='relu'))
    net.add(BatchNormalization())
    net.add(Dropout(0.4))
    # net.add(Flatten())
    net.add(GlobalAveragePooling2D())
    net.add(Dense(512, activation='relu'))
    net.add(Dropout(rate=0.5))
    net.add(Dense(1, activation='sigmoid'))

    save_network_structure(net, version)
    return net


def build_func_cnn(version, input_shape=(200, 100, 1), num_classes=2):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if size == 728 or size == 512:
            x = layers.Dropout(0.5)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # if adding flatten layer reduce number of filters cause its going to be massive
    # x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.Conv2D(1024, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    model = keras.Model(inputs, outputs)
    print("layers")
    print(len(model.layers))

    save_network_structure(model, version)
    return model
