class Model():
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


    def compile(self, optimizer, loss='categorical_crossentalpy', metrics=['accuracy']):
        return

    def fit(self):
        return
