import numpy as np
import random
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

class Activation_Step:
    def forward(self, inputs):
        return (inputs > 0).astype(float)

class Activation_Relu:
    def forward(self, inputs):
        return np.maximum(0, inputs)

class Activation_Linear:
    def forward(self, inputs):
        return inputs

class Activation_Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs



X = np.linspace(0, 2*np.pi, 100).reshape(-1,1)
y = np.sin(X)

dense1 = Layer_Dense (1,8)
dense2 = Layer_Dense (8,8)
dense3 = Layer_Dense (8,1)


Activation_Step = Activation_Step()
Activation_Relu = Activation_Relu()
Activation_Sigmoid = Activation_Sigmoid()


dense1.biases = np.random.randn(1,1)*0.1
dense2.biases = np.random.randn(1,1)*0.2
dense3.biases = np.random.randn(1,1)*0.3

dense1.forward(X)
step_output = Activation_Step.forward(dense1.outputs)
dense2.forward(step_output)
RELU_output = Activation_Relu.forward(dense2.outputs)
outputs = dense3.forward(RELU_output)
sigmoid_output = Activation_Sigmoid.forward(outputs)

print(RELU_output)
print(step_output)
print(sigmoid_output)

plt.plot(X, y, label="True Sine Wave",color="blue")
plt.plot(X,sigmoid_output, label="NN Output",color="red")
plt.legend()
plt.title("Sine Wave Approximation using Neural Network")
plt.show()
