import math
from os import error
from typing import Tuple

import numpy as np
from numpy.core.numeric import argwhere

from proj2_code import fundamental_matrix, two_view_data
from proj2_code.least_squares_fundamental_matrix import solve_F, signed_point_line_errors

def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correctt
    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here
    # http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf slide 

    num_samples = math.log(1 - prob_success) / math.log(1 - ind_prob_correct ** sample_size)

    # raise NotImplementedError
    ##############################

    return num_samples

def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    inliers = None

    ##############################
    # TODO: Student code goes here

    if x_0s.shape[-1] == 2:
        ones = np.ones((x_0s.shape[0], 1))
        x_0s = np.hstack((x_0s, ones))

    if x_1s.shape[-1] == 2:
        ones = np.ones((x_1s.shape[0], 1))
        x_1s = np.hstack((x_1s, ones))
    
    errors = np.abs(np.array(signed_point_line_errors(x_0s, F, x_1s)))
    errors_Fx_1_x_0 = errors[0:errors.shape[0]:2]
    # errors_Ftx_0_x_1 = errors[1:errors.shape[0]:2]
    # errors = (errors_Fx_1_x_0 + errors_Ftx_0_x_1) / 2
    inliers = np.argwhere(errors_Fx_1_x_0 < threshold).flatten()

    # raise NotImplementedError
    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None

    ##############################
    # TODO: Student code goes here

    P = 0.999
    k = 9
    p = 0.90
    best_F = np.random.rand(3, 3)
    inliers_cache = 0

    num_samples = int(calculate_num_ransac_iterations(P, k, p))
    
    for _ in range(num_samples):
        random_samples = np.random.choice(x_0s.shape[0], k, replace=False)
        _x_0s = x_0s[random_samples]
        _x_1s = x_1s[random_samples]

        F = solve_F(_x_0s, _x_1s)
        inliers = find_inliers(x_0s, F, x_1s, 1.0)
        
        if inliers.shape[0] > inliers_cache:
            inliers_cache = inliers.shape[0]
            best_F = F
            inliers_x_0 = x_0s[inliers]
            inliers_x_1 = x_1s[inliers]
            
    # raise NotImplementedError
    ##############################

    return best_F, inliers_x_0, inliers_x_1
