import numpy as np

class Flatten():
    def __init__(self):
        self.type = "flatten"

    def flatten(self, imgs):
        return imgs.flatten()
