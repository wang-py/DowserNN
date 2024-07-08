import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


weights_history = []


class weights_visualization_callback(callbacks.Callback):
    def on_epoch_end(self, batch, logs):
        weights_1, biases_1, weights_2, biases_2 = model.get_weights()
        # print('on_epoch_end() model.weights:', weights_1)
        weights_history.append(weights_1)


if __name__ == "__main__":
    # Generate training and validation data
    X = np.load("test_data/CI_X.npy")
    y = np.load("test_data/CI_y.npy")

    training_N = X.shape[0]  # int(10000)
    X_data = tf.convert_to_tensor(X[:training_N, :])
    y_data = tf.convert_to_tensor(y[:training_N, :])
    input_dim = X_data.shape[1]
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])
    y_test = tf.convert_to_tensor(y[training_N:, :])
    # record weights during each training iteration
    callback = weights_visualization_callback()
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
    history = model.fit(X_data, y_data, epochs=64, batch_size=32,
                        callbacks=callback)
    y_predicted = model.predict(X_data)
    np.set_printoptions(precision=4, suppress=True)
    # print("expected output:\n", y_data)
    # print("predicted output:\n", y_predicted)

    # test with new data
    if N != X_data.shape[0]:
        test_loss = model.evaluate(X_test, y_test)
        print(f"loss: {test_loss:.0%}")
    # print("expected output:\n", y_test)
    # print("predicted output:\n", y_validate)
    # error_percent = np.sum(y_validate - y_test) / np.sum(y_test)

    # Plot training loss
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(history.history['loss'], label='cross entropy')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.set_title('Training Loss')
    # ax.legend()
    fig_a, ax_a = plt.subplots(figsize=(10, 10))
    weights_plot = ax_a.imshow(weights_history[0])
    colorbar = fig_a.colorbar(weights_plot, ax=ax_a, shrink=0.5)
    ax_a.set_title("weights in hidden layer over epochs")
    ax_a.set_xlabel("hidden layer size")
    ax_a.set_ylabel("input dimension")
    epochs = 64
    artists = []
    for i in range(epochs):
        container = ax_a.imshow(weights_history[i])
        epoch_index = ax_a.annotate(f"Epoch = {(i + 1):d}", xy=(0.5, 0.2))
        artists.append([container, epoch_index])

    ani = ArtistAnimation(fig=fig_a, artists=artists, interval=100)
    # ax[1].plot(history.history['cross_entropy'], label='cross entropy')
    # ax[1].set_title('cross entropy')
    # ax[1].legend()
    plt.show()
