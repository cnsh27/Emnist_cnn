import numpy as np

class Conv2d():
    def __init__(self, filterCnt, input_shape, kernel_size=5, kernel_type='edge0'):
        self.filterCnt = filterCnt
        self.row_max = input_shape[0]
        self.col_max = input_shape[1]
        self.kernel_size = kernel_size
        self.kernel = []
        self.filter = []
        # kernel 지정 (identity, edge0, edge1, edge2, sharpen, box_blur, gaussian_blur,emboss)
        
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
            exit('failed')
        # self.result = np.zeros((self.row_max, self.col_max))

    def setImg(self, IMG):
        self.IMG = IMG

    def convolution_unit(self, img, filter, i, j):  # 컨볼루션
        calculate = 0
        for u in range(self.kernel_size):
            for v in range(self.kernel_size):
                calculate += img[i + u][j + v] * filter[u][v]
        
        return np.array([self.relu(calculate)])

    def relu(self, x):
        return max(0, x)

    # TODO : range 의 max 설정이 잘 못 된 것 같다.
    def Convolution2D(self, img, filter):
        result = np.zeros((self.col_max - self.kernel_size + 1, self.row_max - self.kernel_size + 1))
        
        for i in range(self.row_max - self.kernel_size + 1):    
            for j in range(self.col_max - self.kernel_size + 1):
                result[i][j] = self.convolution_unit(img, filter, i, j)
        
        return result

    def layer(self, img):
        nextImgs = []
        
        for i in range(self.filterCnt):
            nextImg = np.zeros(tuple(sum(elem) for elem in zip(img[0].shape, (-4, -4)))) # 
            
            for j in range(img.shape[0]): # 이미지의 채널 수만큼 반복 => 원래는 3번째에 채널 개수이지만 편의를 위해 1번째 인수로 사용
                nextImg = np.array([nextImg] + [self.Convolution2D(img[j], self.filter[i][j])])
                nextImg = nextImg.sum(axis=0)
            
            nextImgs.append(nextImg)
            
        return nextImgs



# 수정사항 - 패딩 없앰. n개의 filter에 따라 n개의 채널을 가지는 이미지 생성.