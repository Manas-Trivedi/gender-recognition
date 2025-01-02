import numpy as np

# we'll be using mean squared erroe for loss calculation
def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

# Optimisation: Stochastic Gradient descent
def sigmoid_deriv(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        #weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedForward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
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
                d_L_d_ypred = -2 * (y_true - y_pred)

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
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = NeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedForward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedForward(frank)) # 0.039 - M
