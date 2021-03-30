import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple

import matplotlib.pyplot as plt


def quantizeRGB(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Quantize the RGB image along all 3 channels and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101
    
    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterColors: k x 3 vector of hue centers
    """
    
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    _origImg = np.reshape(origImg, (-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=101).fit(_origImg)
    labels = kmeans.predict(_origImg)

    clusterCenterColors = kmeans.cluster_centers_

    quantizedImg = np.reshape(clusterCenterColors[labels, :], origImg.shape).astype(np.uint8)

    return quantizedImg, clusterCenterColors
    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    # raise NotImplementedError


def quantizeHSV(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Convert the image to HSV, quantize the Hue channel and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Consider using skimage.color for colorspace conversion
    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101

    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterHues: k x 1 vector of hue centers
    """
    
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    _origImg = rgb2hsv(origImg)
    quantizedImg = np.copy(_origImg)
    
    hue = np.reshape(_origImg[:, :, 0], (-1, 1))
    
    kmeans = KMeans(n_clusters=k, random_state=101).fit(hue)
    labels = kmeans.predict(hue)
    clusterCenterHues = kmeans.cluster_centers_
    
    quantizedImg[:, :, 0] = np.reshape(clusterCenterHues[labels], (_origImg[:, :, 0].shape))
    quantizedImg = (hsv2rgb(quantizedImg) * 255).astype(np.uint8)
   
    return quantizedImg, clusterCenterHues

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    # raise NotImplementedError


def computeQuantizationError(origImg: np.ndarray, quantizedImg: np.ndarray) -> int:
    """
    Calculate the quantization error by finding the sum of squared differences between
    the original and quantized images. Implement a vectorized version (using numpy) of
    this error metric.

    Args:
        - origImg: Original input RGB image with shape H x W x 3 and dtype "uint8"
        - quantizedImg: Image obtained post-quantization with shape H x W x 3 and dtype "uint8"

    Returns
        - error: Quantization error
    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    error = np.float64(origImg) - np.float64(quantizedImg)
    error = np.sum(error ** 2)

    return error

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    # raise NotImplementedError
