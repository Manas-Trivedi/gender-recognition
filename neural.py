import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuralNetwork:
    def __init__(self):
        weights = [0, 1]
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)

        out_o1 = self.o1.feedForward([out_h1, out_h2])
        return out_o1

network = NeuralNetwork()
x = [2, 3]
print(network.feedForward(x))
