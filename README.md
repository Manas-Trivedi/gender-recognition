# Simple Neural Network Implementation

This repository contains a Python implementation of a basic neural network with a focus on binary classification. The network features a simple architecture with two input neurons, two hidden neurons, and one output neuron, using sigmoid activation functions throughout.

## Features

The neural network implementation includes:

- Binary classification capability
- Stochastic Gradient Descent (SGD) optimization
- Binary Cross-Entropy loss function
- Automatic data scaling
- Weight and bias persistence (save/load functionality)
- Configurable learning rate and epochs
- Progress monitoring through epoch loss printing

## Prerequisites

To run this neural network, you'll need:

- Python 3.x
- NumPy
- Pandas

Install the required packages using:

```bash
pip install numpy pandas
```

## Dataset Requirements

The network expects two CSV files:
- `Training set.csv`: For training the model
- `Test set.csv`: For evaluating the model's performance

Each CSV file should have three columns:
- First two columns: Input features
- Third column: Binary target variable (0 or 1)

## Usage

1. Prepare your dataset files according to the format described above.

2. Run the neural network:
```bash
python neural.py
```

3. If previously saved weights exist, you'll be prompted to either:
   - Use existing weights (enter 'y')
   - Initialize new random weights (enter 'n')

The network will automatically:
- Scale your input data
- Train for 1000 epochs (if using new weights)
- Save the trained weights
- Evaluate performance on the test set

## Technical Details

### Network Architecture
- Input Layer: 2 neurons
- Hidden Layer: 2 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation

### Training Parameters
- Learning Rate: 0.01
- Epochs: 1000
- Loss Function: Binary Cross-Entropy
- Optimization: Stochastic Gradient Descent

### Weight Persistence
Weights and biases are saved in:
- `weights.npy`: Contains six weights (w1 through w6)
- `biases.npy`: Contains three biases (b1 through b3)

## Performance Monitoring

The network prints the loss value every 10 epochs during training and provides the final test loss after evaluation. This helps in monitoring the training progress and assessing the model's generalization performance.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.