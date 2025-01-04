import numpy as np
import pandas as pd
import os

# Define dataset
df = pd.read_csv('./Training set.csv')

# Training Dataset
data = df.iloc[:, :2].values  # First two columns as a NumPy array
all_y_trues = df.iloc[:, 2].values  # Third column as a NumPy array

# Testing dataset
tf = pd.read_csv('./Test set.csv')
test_data = tf.iloc[:, :2].values  # First two columns as a NumPy array
test_all_y_trues = tf.iloc[:, 2].values  # Third column as a NumPy array

# Scaling
max_values = np.max(np.vstack((data, test_data)), axis=0)
data /= max_values
test_data /= max_values

# we'll be using mean squared erroe for loss calculation
def binary_cross_entropy_loss(y_true, y_pred):
    # Clip predictions to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Optimisation: Stochastic Gradient descent
def sigmoid_deriv(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        self.existing_biases = False
        # Check if weights exist, otherwise initialize them randomly
        if os.path.exists("weights.npy") and os.path.exists("biases.npy"):
            print("Found saved weights and biases. Do you want to use them? (y/n)")
            response = input()
            if response.lower() == "y":
                self.existing_biases = True
                self.load_weights()
            else:
                self.initialize_weights()
        else:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights and biases randomly if no saved ones exist
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def save_weights(self):
        np.save("weights.npy", [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6])
        np.save("biases.npy", [self.b1, self.b2, self.b3])

    def load_weights(self):
        weights = np.load("weights.npy")
        self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = weights
        biases = np.load("biases.npy")
        self.b1, self.b2, self.b3 = biases

    def feedForward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.01
        epochs = 1000 # loops through the dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                #final value
                y_pred = o1

                #partial derivateives for SGD
                d_L_d_ypred = -(y_true/y_pred - (1-y_true)/(1-y_pred))

                #Neuron o1
                d_ypred_d_w5 = h1 * sigmoid_deriv(sum_o1)
                d_ypred_d_w6 = h2 * sigmoid_deriv(sum_o1)
                d_ypred_d_b3 = sigmoid_deriv(sum_o1)

                d_ypred_d_h1 = self.w5 * sigmoid_deriv(sum_o1)
                d_ypred_d_h2 = self.w6 * sigmoid_deriv(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * sigmoid_deriv(sum_h1)
                d_h1_d_w2 = x[1] * sigmoid_deriv(sum_h1)
                d_h1_d_b1 = sigmoid_deriv(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * sigmoid_deriv(sum_h2)
                d_h2_d_w4 = x[1] * sigmoid_deriv(sum_h2)
                d_h2_d_b2 = sigmoid_deriv(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedForward, 1, data)
                loss = binary_cross_entropy_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

        self.save_weights()

# Train our neural network!
network = NeuralNetwork()
if(network.existing_biases == False):
    network.train(data, all_y_trues)

test_predictions = np.apply_along_axis(network.feedForward, 1, test_data)
test_loss = binary_cross_entropy_loss(test_all_y_trues, test_predictions)
print("Test Loss: %.3f" % test_loss)