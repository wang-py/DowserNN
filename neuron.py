import numpy as np


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Number of inputs
    input_size = 1

    # Number of neurons
    num_neurons = 10

    # Generate random input data
    inputs = np.random.rand(input_size)

    # Initialize weights and biases randomly for each neuron
    weights = np.random.rand(num_neurons, input_size)
    biases = np.random.rand(num_neurons)

    # Compute the weighted sum (z) for each neuron
    z = np.dot(weights, inputs) + biases

    # Apply the activation function to each neuron's weighted sum
    outputs = sigmoid(z)

    # Print the inputs, weights, biases, weighted sums, and outputs
    print("Inputs: ", inputs)
    print("Weights: ", weights)
    print("Biases: ", biases)
    print("Weighted sums (z): ", z)
    print("Outputs (after activation): ", outputs)
