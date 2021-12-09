import numpy as np
import perceptron as pc

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def update(self, params, grads):

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
    def __init__(self, batch):
        self.batch = batch

    def loss(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y+delta) + (1 - t) * np.log(1 - y)) / self.batch


class Layer():
    def __init__(self, perceptronCnt, activation='relu'):
        self.perceptronCnt = perceptronCnt
        self.perceptrons = []
        self.activation = activation
        self.adam = Adam()

    def forward(self):
        xArray = np.empty((0, self.perceptronCnt))
        temp = []
        for perceptron in self.perceptrons:
            temp.append(perceptron)
        for i in range(self.perceptronCnt):
            xArray = np.append(xArray, np.array([temp]), axis=0)

        wArray = np.empty((0, perceptron.wCnt))
        for perceptron in self.perceptrons:
            wArray = np.append(wArray, np.array([perceptron.w]), axis=0)

        zArray = xArray.dot(wArray)

        return zArray
    
    def backward(self):
        pass




    def activation(self, x):
        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'softmax':
            return self.softmax(x)

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        array_x = x - np.max(x)

        exp_x = np.exp(array_x)
        result = exp_x / np.sum(exp_x)
        return result

layer = Layer(5)
layer.perceptrons = [3, 10, 42, 1, 5]

layer.forward()
