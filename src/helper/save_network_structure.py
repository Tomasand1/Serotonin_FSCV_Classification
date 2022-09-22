import os
from helper.create_folder import create_folder
from tensorflow.keras.utils import plot_model


def save_network_structure(net, version):
    summary = net.summary()
    model_path = create_folder('../reports/'+version+"/train/")
    model_name = 'network_structure.png'
    plot_model(net, to_file=os.path.join(
        model_path, model_name), show_shapes=True)

    print(summary)
