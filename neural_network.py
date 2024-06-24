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


def our_function(x) -> float:
    output = np.power(x, 3) + 2 * np.power(x, 2) + 10
    return output

    # output = np.zeros(x.size)
    # output[x > 0.1] = 1


def softmax(xi, xj):
    return np.exp(xi) / np.sum(np.exp(xj))


class hidden_layer:
    def __init__(self, hidden_size, input_dim):
        # weights are randomly assigned
        self.weights = np.random.rand(hidden_size, input_dim) / np.sqrt(input_dim)
        # biases are usually initialized to be 0
        self.biases = np.zeros([hidden_size, 1])
        print("hidden layer weights: ", self.weights)
        print("hidden layer biases: ", self.biases)

    def forward(self, inputs):
        self.X = np.dot(self.weights, inputs) + self.biases
        self.output = relu(self.X)
        self.inputs = inputs
        return self.output

    def dSSRdw(self, dSSR_times_db, output_layer_weights):
        N = self.inputs.shape[0]
        dSSRdw = (output_layer_weights.T.dot(dSSR_times_db) *
                  relu_derivative(self.X)).dot(self.inputs.T)
        dSSRdw /= N
        return dSSRdw

    def dSSRdb(self, dSSR_times_db, output_layer_weights):
        N = dSSR_times_db.shape[0]
        dSSRdb = output_layer_weights.T.dot(dSSR_times_db) * relu_derivative(self.X)
        dSSRdb = np.sum(dSSRdb, axis=1) / N
        return dSSRdb

    def backward(self, dSSR, output_layer_weights, learning_rate):
        step_size_w = self.dSSRdw(dSSR, output_layer_weights) * learning_rate
        # print(f"step_size_w:{step_size_w}")
        self.weights -= step_size_w
        step_size_b = self.dSSRdb(dSSR, output_layer_weights) * learning_rate
        # print(f"step_size_b:{step_size_b}")
        self.biases -= step_size_b.reshape(-1, 1)


class output_layer:
    def __init__(self, hidden_size, output_size):
        self.weights = np.random.rand(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.biases = np.zeros([hidden_size, 1])
        print("output layer weights: ", self.weights)
        print("output layer biases: ", self.biases)

    def forward(self, inputs):
        self.X = np.dot(self.weights, inputs) + self.biases
        self.inputs = inputs
        self.output = relu(self.X)
        return self.output

    def dSSRdw(self, dSSR):
        return dSSR.dot(self.inputs.T)

    def backward(self, dSSR, learning_rate):
        N = dSSR.shape[1]
        # updating the weights of each neuron
        step_size_w = self.dSSRdw(dSSR) / N * learning_rate
        # print(f"step_size_w:{step_size_w}")
        self.weights -= step_size_w
        step_size_b = np.sum(dSSR, axis=1) / N * learning_rate
        # print(f"step_size_b:{step_size_b}")
        self.biases -= step_size_b.reshape(-1, 1)
        return self.weights


# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_size, num_of_data, hidden_size, output_size):
        self.hidden_layer = hidden_layer(hidden_size, input_size)
        self.output_layer = output_layer(hidden_size, hidden_size)

    def forward(self, inputs):
        hidden_layer_output = self.hidden_layer.forward(inputs)
        output_layer_output = self.output_layer.forward(hidden_layer_output)
        return output_layer_output

    def dSSR(self, expected, predicted):
        return -2 * (expected - predicted)

    def train(self, inputs, expected_output, learning_rate=0.01, epochs=5000):
        for epoch in range(epochs):
            # Forward pass
            predicted_output = self.forward(inputs)

            # Backward pass
            dSSR = self.dSSR(expected_output, predicted_output)
            output_layer_weights = self.output_layer.backward(
                    dSSR, learning_rate
                    )
            self.hidden_layer.backward(dSSR,
                                       output_layer_weights.T, learning_rate)

            # Calculate loss
            loss = np.mean((expected_output - predicted_output) ** 2)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


if __name__ == "__main__":
    # Number of inputs
    input_size = 1

    # Output size
    output_size = 1

    # Size of hidden layer
    hidden_size = 4

    # Number of data
    num_of_data = 20

    # Generate random input data
    inputs = np.random.rand(num_of_data * input_size)

    # test output
    # y = np.random.rand(num_of_data)
    y = our_function(inputs).reshape([input_size, num_of_data])

    nn = NeuralNetwork(input_size, num_of_data, hidden_size, output_size)

    nn.train(inputs.reshape([input_size, num_of_data]), y,
             learning_rate=0.001, epochs=10000)

    outputs = nn.forward(inputs.reshape([input_size, num_of_data]))
    print("shape of outputs:", outputs.shape)
    validation_inputs = np.random.rand(num_of_data * input_size)
    validation_outputs = our_function(validation_inputs)
    predicted_outputs = nn.forward(validation_inputs.reshape([input_size, num_of_data]))
    print("original inputs:\n", inputs)
    print("expected output:\n", y)
    print("trained output:\n", outputs)
    print("validation inputs:\n", validation_inputs)
    print("validation outputs:\n", validation_outputs)
    print("trained output with test input:\n", predicted_outputs)
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].scatter(inputs, y, label="original function", color='b')
    ax[0].scatter(inputs, outputs.flatten(), label="trained function", color='r')
    ax[0].legend()

    ax[1].scatter(validation_inputs, validation_outputs, color='b',
                  label="original function using different input")
    ax[1].scatter(validation_inputs, predicted_outputs, color='r',
                  label="trained function using different input")
    ax[1].legend()

    plt.show()
