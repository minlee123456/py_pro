import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weight = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.bias = np.random.uniform(0 , 1, (1, n_neurons))

    def forward(self, inputs):
        self.ouput = np.dot(inputs, self.weight)+self.bias
        return self.ouput

X, y = spiral_data(samples=100, classes=3)

p1 = Layer_Dense(2,3)
output = p1.forward(X)
print(output)

p2 = Layer_Dense(3,5)
output2 = p2.forward(output)
print(output2)