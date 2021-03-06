import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import misc, ndimage
from skimage import feature
from skimage.color import rgb2gray
from matplotlib.patches import Circle
from skimage.feature import canny

def showCircles(
    img: np.ndarray,
    circles: np.ndarray,
    houghAccumulator: np.ndarray,
    showCenter: bool = False,
) -> None:
    """
    Function to plot the identified circles
    and associated centers in the input image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W
        - showCenter: Flag specifying whether to visualize the center
            or not
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    ax1.set_aspect("equal")
    ax1.imshow(img)
    ax2.imshow(houghAccumulator)

    for circle in circles:
        x, y, rad = circle
        circ = Circle((y, x), rad, color="black", fill=False, linewidth=1.5)
        ax1.add_patch(circ)
        if showCenter:
            ax1.scatter(y, x, color="black")
        # ax1.set_title('Center (x,y): (%d, %d) | Radius: %d'%(y, x, rad))
        ax1.set_title('Radius: %d'%(rad))
    plt.show()


def detectCircles(
    img: np.ndarray, radius: int, threshold: float, useGradient: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Implement a hough transform based circle detector that takes as input an
    image, a fixed radius, voting threshold and returns the centers of any detected
    circles of about that size and the hough space used for finding centers.

    NOTE: You are not allowed to use any existing hough transform detector function and
        are expected to implement the circle detection algorithm from scratch. As helper
        functions, you may use 
            - skimage.color.rgb2gray (for RGB to Grayscale conversion)
            - skimage.feature.canny (for edge detection)
            - denoising functions (if required)
        Additionally, you can use the showCircles function defined above to visualize
        the detected circles and the accumulator array.

    NOTE: You may have to tune the "sigma" parameter associated with your edge detector 
        to be able to detect the circles. For debugging, considering visualizing the
        intermediate outputs of your edge detector as well.

    For debugging, you can use im1.jpg to verify your implementation. See if you are able
    to detect circles of radii [75, 90, 100, 150]. Note that your implementation
    will be evaluated on a different image. For the sake of simplicity, you can assume
    that the test image will have the same basic color scheme as the provided image. Any
    hyper-parameters you tune for im1.jpg should also be applicable for the test image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - radius: Radius of circle to be detected
        - threshold: Post-processing threshold to determine circle parameters
            from the accumulator array
        - useGradient: Flag that allows the user to optionally exploit the
            gradient direction measured at edge points.

    Returns:
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W

    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    img_gray = rgb2gray(img)
    sigma = 2
    edges = canny(np.float64(img_gray), sigma=sigma)
    houghAccumulator = np.zeros(edges.shape)

    if useGradient:
        dx = ndimage.sobel(img_gray, axis=0)
        dy = ndimage.sobel(img_gray, axis=1)

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j]:
                if useGradient:
                    theta = np.arctan2(dx[i, j], dy[i, j])
                    a = int(j + radius * np.cos(theta))
                    b = int(i + radius * np.sin(theta))

                    try:
                        houghAccumulator[b][a] += 1
                    except:
                        pass
                    
                    theta = np.arctan2(-dx[i, j],-dy[i, j])
                    a = int(j + radius * np.cos(theta))
                    b = int(i + radius * np.sin(theta))

                    try:
                        houghAccumulator[b][a] += 1
                    except:
                        pass

                else:
                    for theta in np.radians(range(360)):
                        a = int(j - radius * np.cos(theta))
                        b = int(i + radius * np.sin(theta))
                        try:
                            houghAccumulator[b][a] += 1
                        except:
                            pass

    centers = np.where(np.amax(houghAccumulator) * threshold <= houghAccumulator)
    x_centers = np.expand_dims(centers[0], axis=1)
    y_centers = np.expand_dims(centers[1], axis=1)
    radiuses = [radius for _ in range(len(x_centers))]
    radiuses = np.expand_dims(radiuses, axis=1)
    # radiuses = np.expand_dims(houghAccumulator[centers], axis=1)
    circles = np.hstack((x_centers, y_centers, radiuses))

    return circles, houghAccumulator
    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################

    # raise NotImplementedError
