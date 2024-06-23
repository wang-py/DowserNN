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


def generate_input(input_dim, output_dim, num_of_data):
    X = np.random.rand(input_dim, num_of_data)
    y = np.random.rand(output_dim, num_of_data)
    return X, y

    # output = np.zeros(x.size)
    # output[x > 0.1] = 1


def softmax(xi, xj):
    return np.exp(xi) / np.sum(np.exp(xj))

def forward_W1(

if __name__ == "__main__":
    # Number of inputs
    input_size = 2

    # Output size
    output_size = 1

    # Size of hidden layer
    hidden_size = 2

    # Number of data
    num_of_data = 3

    # Generate random input data
    inputs = np.random.rand(num_of_data, input_size)
    # initialilze weights
    W1 = np.random.rand(hidden_size, num_of_data)
    b1 = np.zeros(input_size, 1)

    W2 = np.random.rand(num_of_data, output_size)
    b2 = np.zeros(num_of_data)

    y = our_function(inputs)



    # test output
    # y = np.random.rand(num_of_data)
    # y = our_function(inputs)

    X, y = generate_input(input_size, output_size, num_of_data)


    # validation_inputs = np.random.rand(num_of_data * input_size)
    # jvalidation_outputs = our_function(validation_inputs)
    # predicted_outputs = nn.forward(validation_inputs.reshape([input_size, num_of_data]))
    # print("original inputs:\n", inputs)
    # print("expected output:\n", y)
    # print("trained output:\n", outputs)
    # print("validation inputs:\n", validation_inputs)
    # print("validation outputs:\n", validation_outputs)
    # print("trained output with test input:\n", predicted_outputs)
    # fig, ax = plt.subplots(figsize=(12, 10))
    # ax.scatter(inputs, y, label="original function", color='b')
    # ax.scatter(inputs, outputs.flatten(), label="trained function", color='r')
    # ax.legend()

    # ax[1].scatter(validation_inputs, validation_outputs, color='b',
    #               label="original function using different input")
    # ax[1].scatter(validation_inputs, predicted_outputs, color='r',
    #               label="trained function using different input")
    # ax[1].legend()

    plt.show()
