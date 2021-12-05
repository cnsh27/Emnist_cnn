import numpy as np

class Dense():
    def __init__(self, perceptronCnt, input_shape, reshape, activation='relu'):
        self.perceptronCnt = perceptronCnt
        self.perceptrons = []
        self.activation = activation

    def createPerceptrons():
        return

    def activation(self):
        if self.activation != 'relu':
            self.relu()

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        array_x = x - np.max(x)

        exp_x = np.exp(array_x)
        result = exp_x / np.sum(exp_x)
        return result


dense = Dense(32, input_shape=(28,28,1), activation='softmax')
