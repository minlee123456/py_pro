import numpy as np
import nnfs
from nnfs.datasets import vertical_data

np.random.seed(0)
nnfs.init()  # 랜덤 시드 초기화

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# 데이터셋 로드 (10개의 클래스로 변경)
X, y = vertical_data(samples=100, classes=10)

# 레이어와 활성화 함수 인스턴스화
dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
activation1 = Activation_ReLU()

# dense2의 출력 뉴런 수를 10으로 변경
dense2 = Layer_Dense(n_inputs=3, n_neurons=10)
activation2 = Activation_Softmax()

# 손실 함수 인스턴스화
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = float('inf')
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    # 가중치와 편향에 작은 변화를 더함
    dense1.weights += 0.02 * np.random.randn(2, 3)
    dense1.biases += 0.02 * np.random.randn(1, 3)
    dense2.weights += 0.02 * np.random.randn(3, 10)
    dense2.biases += 0.02 * np.random.randn(1, 10)

    # 순전파
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # 손실 계산
    loss = loss_function.calculate(activation2.output, y)

    # 예측 정확도 계산
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # 손실이 더 낮아지면 가중치 저장
    if loss < lowest_loss:
        print(f'Iteration: {iteration}, Loss: {loss}, Acc: {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()