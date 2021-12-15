
class Perceptron():
    def __init__(self, wCnt=0, threshold=0, maxRepeatCnt = 10):
        self.x = 0
        self.z = 0
        self.wCnt = wCnt
        self.w = [0 for i in range(self.wCnt)]
        self.gradient = 0

    