import numpy as np

class Model():
    model = []
    denses = []
    def __init__(self):
        return

    def add(self, layer):
        if layer.__class__.__name__ == "SingleLayer":
            return
        elif layer.__class__.__name__ == "Dropout":
            return
        elif layer.__class__.__name__ == "Conv2D":
            return
        elif layer.__class__.__name__ == "MaxPooling2D":
            return

    def feedForward(self):
        lastGradients = np.array([])
        lastWeights = np.array([])

        for dense in reversed(self.denses): # 반대로 진행하여 grad를 구하기
            tempLastGradients = np.array([])
            tempLastWeights = np.array([])


            for perceptron in dense.perceptrons:
                tempLastWeights = np.append(tempLastWeights, np.array([perceptron.w]), axis=0)
                grad = lastGradients.dot(np.transpose(lastWeights))

                # relu 함수의 기울기는 x > 0 일때 1, x <= 0 일때 0이다. 따라서 relu함수의 출력값인 z가 0이라면 x <= 0이라는 소리이니, grad도 마찬가지로 0이 된다.
                if perceptron.z == 0:
                    grad = 0

                perceptron.gradient = grad
                tempLastGradients = np.append(tempLastGradients, np.array([grad]), axis=0)
            
            lastWeights = tempLastWeights
            lastGradients = tempLastGradients

    def compile(self, optimizer='adam', loss='categorical_crossentalpy', metrics=['accuracy']):
        pass

    def fit(self, x, y):
        for i in range(len(x)):
            xTrain = x[i]
            yTrain = y[i]
            
            