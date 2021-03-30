import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        self.A = np.load('inputAPS1Q2.npy')
        ###### END CODE HERE ######
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        _A = np.reshape(self.A, (1, -1))
        _A = -np.sort(-_A)
        plt.imshow(_A, aspect='auto', cmap='gray')
        ###### END CODE HERE ######
        pass
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        plt.hist(self.A.ravel(), bins=20, range=(np.min(self.A), np.max(self.A)))
        # plt.savefig('./prob_2_2.png')
        plt.show()
        ###### END CODE HERE ######
        pass
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        ###### START CODE HERE ######
        h, w = self.A.shape
        X = self.A[h//2:h, 0:w//2]
        return X
        ###### END CODE HERE ######
        pass 
    
        ###### return X ###### 
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        ###### START CODE HERE ######
        Y = self.A - np.mean(self.A, axis=None)
        return Y
        ###### END CODE HERE ######
        pass
    
        ###### return Y ######
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100 x 3.
        """
        ###### START CODE HERE ######
        h, w = self.A.shape
        Z = np.zeros((h,w,3), 'float64')
        Z[self.A > np.mean(self.A, axis=None), 0] = 1
        
        # _Z = np.zeros((h,w,3), 'float64')
        # temp = np.copy(self.A)
        # temp[temp > np.mean(self.A, axis=None)] = 1
        # temp[temp <= np.mean(self.A, axis=None)] = 0
        # _Z[:,:,0] += temp
        # print(np.array_equal(_Z,Z))

        plt.imsave('outputZPS1Q2.png', Z)
        plt.imshow(Z)
        
        return Z
        ###### END CODE HERE ######
        pass
    
        ###### return Z ######
        
if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()