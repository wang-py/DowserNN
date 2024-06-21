import numpy as np
from neural_network import our_function
import matplotlib.pyplot as plt


# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Define the InputLayer class
class InputLayer:
    def __init__(self, input_size, hidden_size):
        self.weights = np.random.randn(hidden_size, input_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(hidden_size)

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


# Define the HiddenLayer class
class HiddenLayer:
    def __init__(self, hidden_size, output_size):
        self.weights = np.random.randn(output_size, hidden_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.biases = np.zeros(output_size)

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
        self.input_layer = InputLayer(input_size, hidden_size)
        self.hidden_layer = HiddenLayer(hidden_size, output_size)

    def forward(self, inputs):
        hidden_output = self.input_layer.forward(inputs)
        final_output = self.hidden_layer.forward(hidden_output)
        return final_output

    def train(
        self,
        inputs,
        expected_outputs,
        validation_inputs,
        validation_outputs,
        epochs=1000,
        batch_size=32,
    ):
        for epoch in range(epochs):
            permutation = np.random.permutation(len(inputs))
            inputs_shuffled = inputs[permutation]
            expected_outputs_shuffled = expected_outputs[permutation]

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs_shuffled[i : i + batch_size]
                batch_expected_outputs = expected_outputs_shuffled[i : i + batch_size]

                for x, y in zip(batch_inputs, batch_expected_outputs):
                    # Forward pass
                    actual_output = self.forward(x)

                    # Backward pass
                    hidden_error = self.hidden_layer.backward(y, self.learning_rate)
                    self.input_layer.backward(hidden_error, self.learning_rate)

            # Calculate training loss
            train_loss = np.mean(
                (expected_outputs - np.array([self.forward(x) for x in inputs])) ** 2
            )

            # Calculate validation loss
            val_loss = np.mean(
                (
                    validation_outputs
                    - np.array([self.forward(x) for x in validation_inputs])
                )
                ** 2
            )

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
                )


if __name__ == "__main__":

    # Number of inputs
    input_size = 100

    # Number of neurons in the hidden layer
    hidden_size = 10

    # Number of neurons in the output layer
    output_size = 100

    # Generate random training data
    training_inputs = np.random.rand(1000, input_size)
    training_expected_outputs = our_function(training_inputs)

    # Generate random validation data
    validation_inputs = np.random.rand(200, input_size)
    validation_expected_outputs = our_function(validation_inputs)

    # Create the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    nn.train(
        training_inputs,
        training_expected_outputs,
        validation_inputs,
        validation_expected_outputs,
        epochs=3000,
    )

    # Generate new random input data for prediction
    new_inputs = np.random.rand(input_size)

    # Perform a forward pass with new inputs to make a prediction
    new_outputs = nn.forward(new_inputs)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].scatter(new_inputs, our_function(new_inputs))
    ax[0].set_xlabel("expected function")

    ax[1].scatter(new_inputs[np.where(new_outputs != 0.)], new_outputs[np.where(new_outputs != 0.)])
    ax[1].set_xlabel("trained function")
    plt.show()

    # Print the new inputs and the predicted outputs
    print("New Inputs: ", new_inputs)
    print("Predicted Outputs: ", new_outputs)
