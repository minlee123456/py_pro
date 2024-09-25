import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: 뉴런의 개수
        '''
        self.weights = np.zeros((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation ReLu Class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Activation Softmax Class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


## Loss_CategoricalCrossentropy
class cross_entropy:
    def forward(self, predictions, targets):

        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        #clipl : 범위 지정
        # e = 10, e-7 = 10^-7

        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# 300,2
layer = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()


## forward
layer.forward(X)
activation1.forward(layer.output)
activation2.forward(activation1.output)

# loss calculation
loss_function = cross_entropy()
loss = loss_function.forward(activation2.output, y)

# print loss
print("Categorical Cross-Entropy Loss:", loss)
