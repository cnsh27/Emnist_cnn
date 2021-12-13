import numpy as np

class Flatten():
    def Flatten(self, imgs):
        return np.ravel(imgs, order='C')
