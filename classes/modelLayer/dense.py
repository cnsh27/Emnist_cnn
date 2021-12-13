import numpy as np
import perceptron as pc

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def update(self, params, grads): # params는 w, grads는 w에 대한 손실함수의 기울기 = (예측값 - 실제값) * 입력값

        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= self.lr *self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
        return params

class CCE:
    batch = 1
    lastBatch = 0

    def setBatch(self, batch):
        self.batch = batch

    def loss(self, y, t): # y : 원핫인코딩 된 1차원 배열이 클래스의 크기만큼 길이를 가짐. 따라서 여러 샘플을 가져서 2차원 배열로 존재. t는 실제값
        lastInd = min(self.lastBatch+self.batch, len(y)+1)
        
        yBatch = y[self.lastBatch : lastInd]
        tBatch = t[self.lastBatch : lastInd]
        
        delta = 1e-7
        returnData = -np.sum(tBatch * np.log(yBatch + delta)) / (lastInd - self.lastBatch)

        self.lastBatch += self.batch
        return returnData

    def lossSlope(self, y, t, x):
        return (y - t) * x


class Layer():
    def __init__(self, perceptronCnt, activation='relu'):
        self.perceptronCnt = perceptronCnt
        self.perceptrons = []
        self.activation = activation
        self.adam = Adam()
        self.cce = CCE()

    def forward(self): # 1차원 배열 x와 2차원 배열 w의 합성곱 z
        xArray = np.empty((0, self.perceptronCnt))
        for perceptron in self.perceptrons:
            xArray = np.append(xArray, perceptron.z, axis=0)

        wArray = np.array([])
        for perceptron in self.perceptrons:
            wArray = np.append(wArray, np.array([perceptron.w]), axis=0)

        zArray = xArray.dot(wArray)

        return self.activation(zArray)
    
    def backward(self):
        fixedWArray = np.array([])
        for perceptron in self.perceptrons:
            gradients = []
            fixedWArray = np.append(fixedWArray, np.array([self.adam.update(perceptron.w, gradients)]), axis=0)
        return fixedWArray


    def activation(self, x):
        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'softmax':
            return self.softmax(x)

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x): # x는 일차원 배열
        array_x = x - np.max(x)

        exp_x = np.exp(array_x)
        result = exp_x / np.sum(exp_x)
        return result

layer = Layer(5)
layer.perceptrons = [3, 10, 42, 1, 5]

layer.forward()
