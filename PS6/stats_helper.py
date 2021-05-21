import glob
import os
from os.path import dirname
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    scaler = StandardScaler()
    for f1 in os.listdir(dir_name):
        if '.DS_Store' in f1:
            continue
        for f2 in os.listdir(os.path.join(dir_name, f1)):
            if '.DS_Store' in f2:
                continue
            for f3 in os.listdir(os.path.join(os.path.join(dir_name, f1), f2)):
                if '.DS_Store' in f3:
                    continue
                data_path = os.path.join(os.path.join(os.path.join(dir_name, f1), f2), f3)
                img = Image.open(data_path)
                img = img.convert('L')
                img = np.array(img)
                img = img / 255.0
                img = np.reshape(img, (-1, 1))
                scaler.partial_fit(img)
                
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
