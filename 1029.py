import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()  # 모든 랜덤 변수 초기화


# Dense layer:
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: 출력의 개수/ 뉴런의 개수
        '''
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.ouput = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        '''
        :param dvalues: 뒷 레이어의 미분 전달값
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs
                                            , axis=1, keepdims=True))
        p = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = p

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            # (5,5) -> 25,1
            jacobian_matrix = np.digflat(single_output) - \
                              np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


class Optimizer_SGD:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_rate = self.learning_rate * \
                                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_rate * layer.dweights
            bias_updates = -self.current_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        neg = -np.log(correct_confidences)
        return neg

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class ASLC():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_trues):
        samples = len(dvalues)
        if len(y_trues.shape) == 2:
            y_trues = np.argmax(y_trues, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_trues] -= 1

        self.dinputs = self.dinputs / samples


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = ASLC()

optimizer = Optimizer_SGD(0.1, 5e-7)

for epoch in range(100000):
    dense1.forward(X)
    activation1.forward(dense1.ouput)
    dense2.forward(activation1.outputs)
    loss = loss_activation.forward(dense2.ouput, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch : {epoch}, ' +
              f'acc: {accuracy:.3f},' +
              f'loss: {loss:.3f},' +
              f'lr:{optimizer.current_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()