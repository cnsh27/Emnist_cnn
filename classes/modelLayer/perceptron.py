
class Perceptron():
    def __init__(self, realResult, wCnt=0, threshold=0, maxRepeatCnt = 10):
        self.x = 0
        self.z = 0
        self.wCnt = wCnt
        self.w = list(range(wCnt))
        self.threshold = threshold
        self.maxRepeatCnt = maxRepeatCnt
        self.realResult = realResult
        self.gradient = 0

    