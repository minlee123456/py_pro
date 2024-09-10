import numpy as np


inputs = [[5.0, 2.0, 3.0, 3.5],
          [2.0, 8.0, -1.0, 4.0],
          [-1.5, -1.27, 5.17, 0.87],
          [0.5, -2.27, 0.1, 0.7],
          [2.5, 0.2, -3.7, 1.8]]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, -0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
    ]

weights2 = [
    [0.4, 1.2, 2.4],
    [-3.5, 1.1, 2.6],
    [0.6, -0.27, -0.17]
    ]

biases = [2.0, 3.0, 0.5]
biases2 = [1.0, 2.4, 1.5]

layers_outputs = np.dot(inputs, np.array(weights).T) +biases
layers_outputs2 = np.dot(layers_outputs, np.array(weights2).T) +biases2