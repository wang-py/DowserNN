import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate training and validation data
    X = np.load("test_data/CI_X.npy")
    y = np.load("test_data/CI_y.npy")

    training_N = int(274)
    X_data = tf.convert_to_tensor(X[:training_N, :])
    y_data = tf.convert_to_tensor(y[:training_N, :])
    input_dim = X_data.shape[1]
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])
    y_test = tf.convert_to_tensor(y[training_N:, :])
    # Create a neural network model
    model = Sequential()
    model.add(
        Dense(
            80,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer='zeros'
        )
    )
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy")
    model.build((N, input_dim))

    model.summary()
    # Train the model
    history = model.fit(X_data, y_data, epochs=64, batch_size=32)
    y_predicted = model.predict(X_data)
    np.set_printoptions(precision=4, suppress=True)
    # print("expected output:\n", y_data)
    # print("predicted output:\n", y_predicted)

    # test with new data
    test_loss = model.evaluate(X_test, y_test)
    # print("expected output:\n", y_test)
    # print("predicted output:\n", y_validate)
    # error_percent = np.sum(y_validate - y_test) / np.sum(y_test)
    print(f"loss: {test_loss:.0%}")

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='cross entropy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    # ax[1].plot(history.history['cross_entropy'], label='cross entropy')
    # ax[1].set_title('cross entropy')
    # ax[1].legend()
    plt.show()
