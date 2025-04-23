# Ml-1

import numpy as np

# Input (X) and Output (y) data
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Normalize data
X = X / np.amax(X, axis=0)
y = y / 100

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network architecture
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Training parameters
epochs = 7000
learning_rate = 0.1

# Weight and bias initialization
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, wout) + bout
    predicted_output = sigmoid(final_input)

    # Backpropagation
    error = y - predicted_output
    d_predicted = error * sigmoid_derivative(predicted_output)

    error_hidden = d_predicted.dot(wout.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    wout += hidden_output.T.dot(d_predicted) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    # Optionally update biases
    # bout += np.sum(d_predicted, axis=0, keepdims=True) * learning_rate
    # bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Print results
print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", predicted_output)
