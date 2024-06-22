import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> float:
    return x * (1 - x)


def relu(x) -> float:
    return np.maximum(0, x)


def relu_derivative(x) -> float:
    return np.where(x > 0, 1, 0)


def dSSR(observed, predicted):
    return np.sum(-2 * (observed - predicted))


def our_function(x) -> float:
    return np.power(x, 3) + 2 * np.power(x, 2) + 10


class input_layer:
    def __init__(self, input_size, num_of_data):
        self.weights = np.random.rand(num_of_data, input_size)
        self.biases = np.random.rand(input_size)

    def forward(self, inputs):
        self.outputs = relu(np.dot(self.weights, inputs)) + self.biases
        return self.outputs


class hidden_layer:
    def __init__(self, hidden_size, output_size):
        self.weights = np.random.rand(output_size, hidden_size)
        self.biases = np.random.rand(hidden_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = relu(np.dot(self.weights, inputs) + self.biases)
        return self.output

    def backward(self, expected_output, learning_rate):
        step_size = dSSR(expected_output, self.output) * learning_rate
        self.biases -= step_size


# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.input_layer = input_layer(input_size, hidden_size)
        self.hidden_layer = hidden_layer(hidden_size, output_size)

    def forward(self, inputs):
        input_layer_output = self.input_layer.forward(inputs)
        hidden_layer_output = self.hidden_layer.forward(input_layer_output)
        return hidden_layer_output

    def train(self, inputs, expected_output, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            actual_output = self.forward(inputs)

            # Backward pass
            self.hidden_layer.backward(
                expected_output, self.learning_rate
            )

            # Calculate loss
            loss = np.mean((expected_output - actual_output) ** 2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    # Number of inputs
    input_size = 10

    # Output size
    output_size = 1

    # Size of hidden layer
    hidden_size = 10

    # Number of data
    num_of_data = 100

    # Generate random input data
    inputs = np.random.rand(num_of_data * input_size)

    # test output
    y = our_function(inputs)

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    nn.train(inputs.reshape([num_of_data, input_size]), y, epochs=2000)

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
