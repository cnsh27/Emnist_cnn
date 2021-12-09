import numpy as np

class Maxpooling():
    def __init__(self, IMG, pool_size=(2,2)):
        self.IMG = IMG
        self.pool_size = pool_size
    
    def maxpooling(self):
        row_max = self.pool_size[0]
        col_max = self.pool_size[1]
        result = np.zeros((row_max, col_max))

        for i in range(len(self.IMG) - (row_max - 1)):
            for j in range(len(self.IMG[0])- (col_max - 1)):
                maximum = 0
                for u in range(row_max):
                    for v in range(col_max):
                        if maximum <= self.IMG[i+u][j+v]:
                            maximum = self.IMG[i+u][j+v]
                result[i][j] = maximum
       
        return result

