import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######
        self.img = cv2.imread('inputPS1Q3.jpg')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        ###### END CODE HERE ######
        pass
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # return gray
        
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
        ###### END CODE HERE ######
        pass
    
        ###### return gray ######
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        swapImg = np.copy(self.img)
        red = np.copy(self.img[:,:,0])
        green = np.copy(self.img[:,:,1])
        
        swapImg[:,:,0] = green
        swapImg[:,:,1] = red
        
        # plt.imshow(swapImg)
        # cv2.imwrite('./prob_3_1.png', swapImg)

        return swapImg
    
        ###### END CODE HERE ######
        pass
    
        ###### return swapImg ######
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = np.array(self.rgb2gray(self.img), dtype=int)
        # plt.imshow(grayImg)
        # cv2.imwrite('./prob_3_2.png', grayImg)
        
        return grayImg
        ###### END CODE HERE ######
        pass
    
        ###### return grayImg ######
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        negativeImg = np.ones(self.prob_3_2().shape, dtype=int) * 255
        negativeImg = negativeImg - self.prob_3_2()
        # plt.imshow(negativeImg, cmap='gray')
        # cv2.imwrite('./prob_3_3.png', negativeImg)
        
        return negativeImg

        ###### END CODE HERE ######
        pass
    
        ###### return negativeImg ######
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        mirrorImg = cv2.flip(self.prob_3_2(), 1)
        # plt.imshow(mirrorImg, cmap='gray')
        # cv2.imwrite('./prob_3_4.png', mirrorImg)
        
        return mirrorImg
        ###### END CODE HERE ######
        pass
    
        ###### return mirrorImg ######
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        
        grayImg = self.prob_3_2()
        mirrorImg = self.prob_3_4()

        _grayImg = np.array(grayImg, dtype=float)
        _mirrorImg = np.array(mirrorImg, dtype=float)
        avgImg = (_grayImg + _mirrorImg) / 2
        avgImg = np.array(avgImg, dtype=int)
        # plt.imshow(avgImg, cmap='gray')
        # cv2.imwrite('./prob_3_5.png', avgImg)
        
        return avgImg

        ###### END CODE HERE ######
        pass
    
        ###### return avgImg ######
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        
        # N = np.random.randint(low=0, high=256, size=grayImg.shape)
        # np.save('./noise.npy', N)
        
        N = np.load('noise.npy')
        
        _grayImg = np.array(grayImg, dtype=float)
        _N = np.array(N, dtype=float)

        _addNoiseImg = _grayImg + _N
        _addNoiseImg = np.clip(_addNoiseImg, a_min=0, a_max=255)
        addNoiseImg = np.array(_addNoiseImg, dtype=int)
        # plt.imshow(addNoiseImg, cmap='gray')
        # cv2.imwrite('./prob_3_6.png', addNoiseImg)
        return addNoiseImg

        ###### END CODE HERE ######
        pass
    
        ###### return addNoiseImg ######
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()