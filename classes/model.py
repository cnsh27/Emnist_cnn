import numpy as np

class Model():
    models = []
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

    def fit(self, x, y, batch_size):
        self.batch_size = batch_size

        batchDatas = np.array([])

        for i in range(0, len(x), self.batch_size):
            datas = np.array([])
            for j in range(i, min(i + self.batch_size + 1, len(x))):
                xTrain = np.array([1, x[i]])
                yTrain = y[i]
                data = xTrain
                for model in self.models:
                    if model.type == "conv2d":
                        data = model.layer(data)
                    elif model.type == "maxpooling":
                        data = model.maxpooling(data)
                    elif model.type == "flatten":
                        data = model.flatten(data)
                        datas = np.append(datas, data, axis=0)
            self.fitDense(datas)

    def fitDense(self, datas): # 다층신경망 전용 fit
        for i in range(len(datas)):
            data = datas[i]
            w = []
            for dense in self.denses:
                for i in range(len(dense.perceptrons)):
                    dense.perceptrons[i].x = data[i]
                data = dense.forward()
                self.feedForward()
            for dense in reversed(self.denses):
                for i in range(len(dense.perceptrons)):
                    dense.perceptrons[i].w = w[i]
                w = dense.backward()
                

                
