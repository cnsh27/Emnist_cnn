import numpy as np

class Conv2d():
    def __init__(self, filters, IMG, input_shape, kernel_type='edge0'):
        self.row_max = input_shape[0]
        self.col_max = input_shape[1]
        # kernel 지정 (identity, edge0, edge1, edge2, sharpen, box_blur, gaussian_blur,emboss)
        self.filters = filters
        if kernel_type == 'identity':
            self.kernel = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]])
        elif kernel_type == 'edge0':
            self.kernel = np.array([[1, 0, -1],
                                    [0, 0, 0],
                                    [-1, 0, 1]])
        elif kernel_type == 'edge1':
            self.kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

        elif kernel_type == 'edge2':
            self.kernel = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]])

        elif kernel_type == 'sharpen':
            self.kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])

        elif kernel_type == 'box_blur':
            self.kernel = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]]) * 0.111

        elif kernel_type == 'gaussian_blur':
            self.kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) * 0.0625

        elif kernel_type == 'emboss':
            self.kernel = np.array([[-2, -1, 0],
                                    [-1, 1, 1],
                                    [0, 1, 2]])
        else:
            exit('failed upload kernel_type!')

        self.IMG = IMG
        # self.result = np.zeros((self.row_max, self.col_max))

    def pad_with(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def convolution_unit(self, i, j):  # 컨볼루션
        calculate = 0
        for u in range(3):
            for v in range(3):
                calculate += self.IMG[i + u][j + v] * self.kernel[u][v]
        if calculate < 0:
            return 0  # activation function RELU
        else:
            return calculate

    # TODO : range 의 max 설정이 잘 못 된 것 같다.
    def Convolution2D(self):
        for i in range(self.row_max):
            for j in range(self.col_max):
                self.result[i][j] = self.convolution_unit(i, j)

    def layer(self):
        for k in range(self.filters):
            self.IMG = np.pad(self.IMG, 1, self.pad_with)  # padding
            self.result = np.zeros((self.row_max, self.col_max))
            self.Convolution2D()
            self.IMG = self.result
        return self.IMG

