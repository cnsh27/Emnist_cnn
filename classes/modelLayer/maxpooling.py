import numpy as np

class Maxpooling(): # stride = 2로 고정
    def __init__(self, pool_size=(2,2)):
        self.pool_size = pool_size
        self.type = "maxpooling"
    
    def maxpooling(self, imgs):
        newImgs = np.array([])
        for i in range(len(imgs)):
            newImg = np.zeros((imgs[i].shape[0]/2, imgs[i].shape[1]/2))
            for y in range(0, imgs.shape[1], 2):
                for x in range(0, imgs.shape[2], 2):
                    maximum = 0
                    for dy in range(self.pool_size[0]):
                        for dx in range(self.pool_size[1]):
                            maximum = max(maximum, imgs[i][y+dy][x+dx])
                    newImg[y/2][x/2] = maximum
            newImgs = np.append(newImgs, newImg, axis=0)
       
        return newImgs

