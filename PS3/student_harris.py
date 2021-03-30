import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    # raise NotImplementedError('`get_gaussian_kernel` function in ' +
    # '`student_harris.py` needs to be implemented')

    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.dot(kernel_1d, kernel_1d.T)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filt):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter

    Note: It is okay to use nested for loops to implement this
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################

    # raise NotImplementedError('`my_filter2D` function in ' +
    # '`student_harris.py` needs to be implemented')

    filt = np.flipud(np.fliplr(filt))
    conv_image = np.zeros_like(image)
    
    padded_image = cv2.copyMakeBorder(image, top=filt.shape[0]//2, bottom=filt.shape[0]//2,
                                        left=filt.shape[1]//2, right=filt.shape[1]//2,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    for j in range(conv_image.shape[1]):
        for i in range(conv_image.shape[0]):
            conv_image[i, j] = (filt * padded_image[i:i+filt.shape[0], j:j+filt.shape[1]]).sum()

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    # raise NotImplementedError('`get_gradients` function in ' +
    # '`student_harris.py` needs to be implemented')

    sobel_x = np.array(
        [
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ])
    sobel_y = np.array(
        [
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ])

    ix = my_filter2D(image, sobel_x)
    iy = my_filter2D(image, sobel_y)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    # raise NotImplementedError('`remove_border_vals` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    
    wd = int(window_size // 2)
    idx = np.where(y - wd >= 0)
    x = x[idx]
    y = y[idx]
    c = c[idx]

    idx = np.where(x - wd >= 0)
    x = x[idx]
    y = y[idx]
    c = c[idx]

    idx = np.where(y + wd < image.shape[0])
    x = x[idx]
    y = y[idx]
    c = c[idx]

    idx = np.where(x + wd < image.shape[1])
    x = x[idx]
    y = y[idx]
    c = c[idx]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.
    (Refer to Eq 4.8 in Szeliski Sec 4.1.1 for exact equations)
    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with ix*ix
    -   sy2: A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with iy*iy
    -   sxsy: (optional): A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with ix*iy
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    # raise NotImplementedError('`second_moments` function in ' +
    # '`student_harris.py` needs to be implemented')

    gk = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix**2, gk)
    sy2 = my_filter2D(iy**2, gk)
    sxsy = my_filter2D(ix*iy, gk)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    # raise NotImplementedError('`corner_response` function in ' +
    # '`student_harris.py` needs to be implemented')

    trace = sx2 + sy2
    R = (sx2 * sy2 - sxsy**2) -  alpha * (trace**2)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the global median.
    
    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    # raise NotImplementedError('`non_max_suppression` function in ' +
    # '`student_harris.py` needs to be implemented')median = np.median(R)

    zeros = np.zeros_like(R)
    median = np.median(R)
    thresholded = np.where(R >= median, R, zeros) # remove all values that are below the global median
    mf = maximum_filter(thresholded, neighborhood_size) # filtered array & has the same shape as input
    R_local_pts = np.where(mf == R, R, zeros)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.


    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None
    

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################
    
    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    # print(image.shape)
    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)

    R = corner_response(sx2, sy2, sxsy, 0.05)
    # print(R.shape)
    R_local_pts = non_max_suppression(R)
    # print(R_local_pts.shape)

    idx = np.argsort(-1 * R_local_pts.flatten())
    x = idx % R.shape[1]
    y = idx // R.shape[1]
    confidences = np.sort(-1 * R_local_pts.flatten())

    x, y, confidences = remove_border_vals(image, x, y, confidences)
    x = x[:n_pts]
    y = y[:n_pts]
    confidences = confidences[:n_pts]

    # i = np.argsort(-1 * R_local_pts, axis=None, kind='mergesort')
    # j = np.unravel_index(i, R_local_pts.shape)
    # R_idx = np.vstack(j).T
    # print(R_idx.shape)

    # print(x.shape)
    # print(y.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x,y, R_local_pts, confidences


