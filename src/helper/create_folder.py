import os
from config.definitions import ROOT_DIR


def create_folder(path):
    """Create folder

    Args:
        path (string): data path

    Returns:
     string: new path
    """
    new_path = os.path.join(ROOT_DIR, path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path
