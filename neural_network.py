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


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def calculate_gradient_SSR(Z1, A1, Z2, A2, W2, X, dSSR):
    N = X.shape[1]
    dSSRdW2 = dSSR.dot(A1.T)
    dW2 = dSSRdW2 / N
    db2 = np.sum(dSSR, axis=1) / N
    dSSRdW1 = (W2.T.dot(dSSR) * relu_derivative(Z1)).dot(X.T)
    dW1 = dSSRdW1 / N
    dSSRdb1 = W2.T.dot(dSSR) * relu_derivative(Z1)
    db1 = np.sum(dSSRdb1, axis=1) / N

    return dW1, db1, dW2, db2


def calculate_gradient_CE(Z1, A1, Z2, A2, W2, X, delta):
    N = X.shape[1]
    dCEdW2 = delta.dot(A1.T)
    dW2 = dCEdW2 / N
    db2 = np.sum(delta, axis=1) / N
    dCEdW1 = W2.T.dot(delta) * relu_derivative(Z1)
    dW1 = dCEdW1.dot(X.T) / N
    dCEdb1 = np.sum(dCEdW1)
    db1 = dCEdb1 / N

    return dW1, db1, dW2, db2


def gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, step_size):
    W1 -= dW1 * step_size
    b1 -= db1.reshape(-1, 1) * step_size
    W2 -= dW2 * step_size
    b2 -= db2.reshape(-1, 1) * step_size

    return W1, b1, W2, b2


def get_dSSR(expected, predicted):
    return -2 * (expected - predicted)


def get_SSR(expected, predicted):
    return np.sum(np.square(expected - predicted))


def get_total_cross_entropy(A2):
    return np.sum(-np.log(A2))


def get_cross_entropy(A2):
    return np.sum(-np.log(A2), axis=0)


def get_delta(y, A2):
    return A2 - y


def train(W1, b1, W2, b2, X, y, step_size, iters):
    # y_one_hot_vec = np.ones(y.shape)
    loss = np.zeros(iters)
    cross_entropy = np.zeros(iters)
    for i in range(iters):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        # argmax_index = np.argmax(A2)
        # output = A2[argmax_index]
        # print(f"argmax_index: {argmax_index}")
        total_cross_entropy = get_total_cross_entropy(A2)
        delta = get_delta(y, A2)
        SSR = get_SSR(y, A2)
        loss[i] = SSR
        cross_entropy[i] = total_cross_entropy
        if i % 100 == 0:
            print(f"cross entropy at {i}: {total_cross_entropy}")
            print(f"SSR at {i}: {SSR}")
        # dSSR = get_dSSR(y, A2)
        # print(f"dSSR: {dSSR}")
        dW1, db1, dW2, db2 = calculate_gradient_CE(Z1, A1, Z2, A2, W2, X, delta)
        # print(f"dW1: {dW1} db1: {db1} dW2: {dW2} db1: {db2}")
        W1, b2, W2, b2 = gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2,
                                          step_size)
        # print(f"W1: {W1} b1: {b1} W2: {W2} b1: {b2}")

    return W1, b1, W2, b2, A2, loss, cross_entropy


def output(argmax_vec, A2):
    return A2[argmax_vec]


if __name__ == "__main__":
    # Number of inputs
    input_dim = 80

    # Output size
    output_size = 2

    # Size of hidden layer
    hidden_size = 80

    # Number of data
    num_of_data = 6

    # Generate random input data
    inputs = np.random.rand(input_dim * num_of_data) - 0.5
    # initialilze weights
    W1 = np.random.rand(hidden_size, input_dim) / np.sqrt(input_dim)
    b1 = np.zeros([hidden_size, 1])

    W2 = np.random.rand(output_size, hidden_size) / np.sqrt(hidden_size)
    b2 = np.zeros([output_size, 1])

    y = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]).T
    inputs_reshaped = inputs.reshape([input_dim, num_of_data])
    y_reshaped = y.reshape([-1, 1])
    np.set_printoptions(precision=3, suppress=True)
    W1, b1, W2, b2, A2, loss, cross_entropy = train(W1, b1, W2, b2,
                                                    inputs_reshaped, y,
                                                    step_size=0.01,
                                                    iters=10000)

    # test output
    A2 = forward(W1, b1, W2, b2, inputs.reshape([input_dim, num_of_data]))[3]
    # print(f"original inputs: {inputs}")
    print(f"expected output:\n{y}")
    print(f"trained output:\n{A2}")

    # validation_inputs = np.random.rand(num_of_data * input_dim) - 0.5
    # validation_outputs = our_function(validation_inputs)
    # predicted_outputs = forward(W1, b1, W2, b2, validation_inputs.reshape([input_dim, num_of_data]))[3]
    # print("original inputs:\n", inputs_reshaped)
    # print("expected output:\n", y)
    # print("trained output:\n", A2)
    # print("validation inputs:\n", validation_inputs)
    # print("validation outputs:\n", validation_outputs)
    # print("trained output with test input:\n", predicted_outputs)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(10000), loss, label="loss function", color='b')
    ax.plot(np.arange(10000), cross_entropy, label="cross entropy", color='r')
    ax.legend()

    # ax[1].scatter(validation_inputs, validation_outputs, color='b',
    #               label="original function using different input")
    # ax[1].scatter(validation_inputs, predicted_outputs.flatten(), color='r',
    #               label="trained function using different input")
    # ax[1].legend()

    plt.show()
