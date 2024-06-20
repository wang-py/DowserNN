import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> float:
    return x * (1 - x)


def relu(x) -> float:
    return np.maximum(0, x)


def relu_derivative(x) -> float:
    return np.where(x > 0, 1, 0)


def SSR(input_data, output):
    return np.sum(np.square(output - input_data))


def our_function(x) -> float:
    return np.power(x, 3) + 2 * np.power(x, 2) + 10


class input_layer:
    def __init__(self, input_size, num_neurons):
        self.weights = np.random.rand(num_neurons, input_size)
        self.biases = np.random.rand(num_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(self.weights, inputs) + self.biases)
        return self.output

    def backward(self, output_error, learning_rate):
        error = output_error * sigmoid_derivative(self.output)
        weights_update = np.dot(error.reshape(-1, 1), self.inputs.reshape(1, -1))
        self.weights += learning_rate * weights_update
        self.biases += learning_rate * error
        return np.dot(self.weights.T, error)


class hidden_layer:
    def __init__(self, input_size, num_neurons):
        self.weights = np.random.rand(num_neurons, input_size)
        self.biases = np.random.rand(num_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = relu(np.dot(self.weights, inputs) + self.biases)
        return self.output

    def backward(self, expected_output, learning_rate):
        error = expected_output - self.output
        delta = error * relu_derivative(self.output)
        weights_update = np.dot(delta.reshape(-1, 1), self.inputs.reshape(1, -1))
        self.weights += learning_rate * weights_update
        self.biases += learning_rate * delta
        return np.dot(self.weights.T, delta)


# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.input_layer = input_layer(input_size, hidden_size)
        self.hidden_layer = hidden_layer(hidden_size, output_size)

    def forward(self, inputs):
        hidden_output = self.input_layer.forward(inputs)
        final_output = self.hidden_layer.forward(hidden_output)
        return final_output

    def train(self, inputs, expected_output, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            actual_output = self.forward(inputs)

            # Backward pass
            hidden_error = self.hidden_layer.backward(
                expected_output, self.learning_rate
            )
            self.input_layer.backward(hidden_error, self.learning_rate)

            # Calculate loss
            loss = np.mean((expected_output - actual_output) ** 2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    # Number of inputs
    input_size = 100

    # Output size
    output_size = 100

    # Size of hidden layer
    hidden_size = 10
=======


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.rand(hidden_size, input_size)
        self.biases_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(output_size, hidden_size)
        self.biases_output = np.random.rand(output_size)

    def forward(self, inputs):
        # Forward pass through hidden layer
        hidden_input = np.dot(self.weights_input_hidden, inputs) + self.biases_hidden
        hidden_output = relu(hidden_input)

        # Forward pass through output layer
        final_input = (
            np.dot(self.weights_hidden_output, hidden_output) + self.biases_output
        )
        final_output = sigmoid(final_input)

        return final_output


if __name__ == "__main__":
    # Number of inputs
    input_size = 1

    # Number of neurons
    num_neurons = 10
>>>>>>> 955efe0 (added hidden layer)

    # Generate random input data
    inputs = np.random.rand(input_size)

<<<<<<< HEAD
    # test output
    y = our_function(inputs)

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    nn.train(inputs, y, epochs=1000)

    outputs = nn.forward(inputs)
    test_inputs = np.random.rand(input_size)
    test_outputs = our_function(test_inputs)
    predicted_outputs = nn.forward(test_inputs)
    fig, ax = plt.subplots(4, 1, figsize=(12, 15))
    ax[0].scatter(inputs, y)
    ax[0].set_xlabel("original function")

    ax[1].scatter(inputs, outputs)
    ax[1].set_xlabel("trained function")

    ax[2].scatter(test_inputs, test_outputs)
    ax[2].set_xlabel("original function using different input")

    ax[3].scatter(test_inputs, predicted_outputs)
    ax[3].set_xlabel("trained function using different input")
    print("original inputs:\n", inputs)
    print("expected output:\n", y)
    print("trained output:\n", outputs)
    print("test inputs:\n", test_inputs)
    print("test outputs:\n", test_outputs)
    print("trained output with test input:\n", predicted_outputs)
    plt.show()
=======
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
>>>>>>> 955efe0 (added hidden layer)
