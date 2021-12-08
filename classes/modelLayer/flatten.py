import numpy as np

class Flatten():
    def __init__(self, IMG, input_shape):
        self.row_max = input_shape[0]
        self.col_max = input_shape[1]
        self.IMG = IMG
        return
    
    def Flatten(self):
        size = self.row_max * self.col_max
        return np.reshape(self.IMG, size)
