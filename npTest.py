import numpy as np
from numpy.core.records import array
result = [
    [3, 4, 5],
    [2, 6, 10],
    [2, 4, 92]
]
result = np.array(result)
sumA = [result] + [result]
sumA = np.array(sumA)
print(sumA.sum(axis=0))