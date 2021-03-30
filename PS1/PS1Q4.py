import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######
        self.indoor = cv2.imread('./indoor.png')
        self.indoor = cv2.cvtColor(self.indoor, cv2.COLOR_BGR2RGB)
        
        self.outdoor = cv2.imread('./outdoor.png')
        self.outdoor = cv2.cvtColor(self.outdoor, cv2.COLOR_BGR2RGB)

        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######
        indoorLAB = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        outdoorLAB = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)

        columns = 3
        rows = 2
        
        rgbfig = plt.figure(num=1, figsize=(10, 8))
        for i in range(columns):
            _indoorRGB = rgbfig.add_subplot(rows, columns, i+1)
            _indoorRGB.title.set_text('indoor\n' + 'RGB'[i])
            plt.imshow(self.indoor[:,:,i], 'gray')
            plt.axis('off')

            _outdoorRGB = rgbfig.add_subplot(rows, columns, i+4)
            _outdoorRGB.title.set_text('outdoor\n' + 'RGB'[i])
            plt.imshow(self.outdoor[:,:,i], 'gray')
            plt.axis('off')
        
        # plt.savefig('./prob_4_1_rgb.png')
        plt.show()
        
        labfig = plt.figure(num=2, figsize=(10, 8))
        for i in range(columns):
            _indoorLAB = labfig.add_subplot(rows, columns, i+1)
            _indoorLAB.title.set_text('indoor\n' + 'LAB'[i])
            plt.imshow(indoorLAB[:,:,i], 'gray')
            plt.axis('off')

            _outdoorLAB = labfig.add_subplot(rows, columns, i+4)
            _outdoorLAB.title.set_text('outdoor\n' + 'LAB'[i])
            plt.imshow(outdoorLAB[:,:,i], 'gray')
            plt.axis('off')
        
        # plt.savefig('./prob_4_1_lab.png')
        plt.show()
        
        ###### END CODE HERE ######
        pass

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        img = io.imread('inputPS1Q4.jpg')
        img = img / 255.0 #  typecasting (double) to transform the image to the [0,1]
        HSV = np.empty_like(img)
        
        ###### START CODE HERE ######
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                R = img[i,j,0]
                G = img[i,j,1]
                B = img[i,j,2]
                
                V = max(R,G,B) # preception of brightness = value
                m = min(R,G,B)
                C = V - m

                HSV[i,j,2] = V
                
                if V == 0:
                    HSV[i,j,1] = 0
                else:
                    HSV[i,j,1] = C/V # the intensity of the color compared to white = saturatio
                    
                # hue computation
                if C == 0:
                    HSV[i,j,0] = 0
                elif V == R:
                    HSV[i,j,0] = (G-B)/C
                elif V == G:
                    HSV[i,j,0] = (B-R)/C + 2.0
                elif V == B:
                    HSV[i,j,0] = (R-G)/C + 4.0
                
                if HSV[i,j,0] < 0:
                    HSV[i,j,0] = HSV[i,j,0]/6.0 + 1.0
                else:
                    HSV[i,j,0] /= 6.0
                
        # _img = cv2.imread('inputPS1Q4.jpg')
        # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
        # plt.imshow(_img)
        # cv2.imwrite('./outputPS1Q4.png', _img)
        # print(np.array_equal(_img,HSV))
        
        plt.imsave('outputPS1Q4.png', HSV)
        plt.imshow(HSV)
        return HSV

        ###### END CODE HERE ######
        pass
    
        ###### return HSV ######
        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_2()
