
class Perceptron():
    def __init__(self, realResult, wCnt=0, threshold=0, maxRepeatCnt = 10):
        self.wCnt = wCnt
        self.w = list(range(wCnt))
        self.threshold = threshold
        self.maxRepeatCnt = maxRepeatCnt
        self.realResult = realResult

    def fit(self, trainData):
        for i in range(self.maxRepeatCnt):
            netInput = self.getNetInput(trainData)

            target = self.relu(self.realResult)
            predict = self.relu(netInput)

            if target != predict:
                for j in range(self.wCnt):
                    self.w[j] += learningRate * trainData[j] * (target - predict)
            
    def getNetInput(self, trainData):
        sum = 0
        for i in range(len(trainData)):
            if i < len(self.w):
                sum += trainData[i] * self.w[i]
            else:
                print("퍼셉트론의 가중치 개수가 trainData 개수보다 작습니다. trainData count =", len(trainData), "wCnt =", self.wCnt)
                return False
        return sum

    def relu(self, x):
        if x < 0:
            return 0
        else:
            return x