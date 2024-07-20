import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


class weights_visualization_callback(callbacks.Callback):
    def __init__(self, num_of_layers):
        self.weights_history = [[] for _ in range(num_of_layers)]
        self.weights_index = np.arange(0, num_of_layers * 2, 2)
        self.num_of_layers = num_of_layers

    def on_epoch_end(self, batch, logs):
        weights_biases = model.get_weights()
        # print('on_epoch_end() model.weights:', weights_1)

        for i in range(self.num_of_layers):
            cur_weight_i = self.weights_index[i]
            self.weights_history[i].append(weights_biases[cur_weight_i])

    def get_weights(self):
        return np.array(self.weights_history)


def get_model_accuracy(y_expected, y_predicted):
    assert y_expected.shape[0] == y_predicted.shape[0]
    y_expected = np.array(y_expected)
    y_predicted = np.array(y_predicted)
    accuracy_values = np.zeros(y_expected.shape[0])
    for i in range(accuracy_values.shape[0]):
        accuracy_values[i] = y_predicted[i].dot(y_expected[i].T)

    return np.sort(accuracy_values)


if __name__ == "__main__":
    # Generate training and validation data
    X_file = sys.argv[1]
    y_file = sys.argv[2]
    X_yes_file = sys.argv[3]
    y_yes_file = sys.argv[4]
    X = np.load(X_file)
    y = np.load(y_file)
    X_yes = np.load(X_yes_file)
    y_yes = np.load(y_yes_file)

    training_N = int(X.shape[0])  # int(33000)
    X_data = tf.convert_to_tensor(X[:training_N, :])
    y_data = tf.convert_to_tensor(y[:training_N, :])
    X_validate = tf.convert_to_tensor(X_yes)
    y_validate = tf.convert_to_tensor(y_yes)
    input_dim = X_data.shape[1]
    hidden_dim = 128
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])
    y_test = tf.convert_to_tensor(y[training_N:, :])
    # record weights during each training iteration
    callback = weights_visualization_callback(num_of_layers=3)
    # Create a neural network model
    model = Sequential()
    model.add(
        Dense(
            hidden_dim,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer='zeros'
        )
    )
    model.add(
        Dense(
            40,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer='zeros'
        )
    )
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss="binary_crossentropy", metrics=['accuracy'])
    model.build((N, input_dim))

    model.summary()
    epochs = 100
    # Train the model
    history = model.fit(X_data, y_data, epochs=epochs, batch_size=32,
                        callbacks=callback)
    np.set_printoptions(precision=4, suppress=True)
    # print("expected output:\n", y_data)
    # print("predicted output:\n", y_predicted)

    # test with new data
    training_loss, training_accuracy = model.evaluate(X_data, y_data)
    print(f"training loss: {training_loss}")
    test_loss = None
    if training_N != X.shape[0]:
        test_loss, accuracy = model.evaluate(X_test, y_test)
        print(f"test loss: {test_loss}")  # , test accuracy: {accuracy:.2%}")
    # print("expected output:\n", y_test)
    # print("predicted output:\n", y_validate)
    # error_percent = np.sum(y_validate - y_test) / np.sum(y_test)

    # plot training loss
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='cross entropy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.axhline(training_loss, color='b', linestyle='--',
               label='training cross entropy')
    if test_loss is not None:
        ax.axhline(test_loss, color='r', linestyle='--',
                   label='test cross entropy')
    ax.set_title('Training Loss')
    ax.legend()

    # plot confidence for water molecules
    y_predicted = model.predict(X_validate)
    accuracy_values = get_model_accuracy(y_validate, y_predicted)
    accuracy_threshold = 0.9
    num_above_threshold = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold = num_above_threshold / num_of_water
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(num_of_water), accuracy_values)
    ax.axhline(accuracy_threshold, color='k', linestyle='--',
               label=f'accuracy threshold = {accuracy_threshold}\n' +
               f'% water above threshold: {percent_above_threshold:.0%}')
    ax.set_xlabel("water index")
    ax.set_ylabel("confidence")
    ax.legend()

    # visualizing weights
    weights_history = callback.get_weights()
    fig_a, ax_a = plt.subplots(subplot_kw={"projection": "3d"},
                               figsize=(12, 12))
    plot_X = np.arange(hidden_dim)
    plot_Y = np.arange(input_dim)
    plot_X, plot_Y = np.meshgrid(plot_X, plot_Y)
    v_min = np.array(weights_history).min()
    v_max = np.array(weights_history).max()
    print(f"minimum weight: {v_min:.2f}")
    print(f"maximum weight: {v_max:.2f}")
    weights_surf = ax_a.plot_surface(plot_X, plot_Y, weights_history[0],
                                     cmap='hot',
                                     vmin=v_min, vmax=v_max)
    # weights_plot = ax_a.imshow(weights_history[0], cmap='hot',
    #                            vmin=v_min, vmax=v_max)
    colorbar = fig_a.colorbar(weights_surf, ax=ax_a, shrink=0.5)
    ax_a.set_title("weights in hidden layer over epochs")
    ax_a.set_xlabel("hidden layer size")
    ax_a.set_ylabel("input dimension")
    artists = []
    for i in range(epochs):
        # container = ax_a.imshow(weights_history[i], cmap='hot')
        container = ax_a.plot_surface(plot_X, plot_Y, weights_history[i],
                                      cmap='hot')
        epoch_index = ax_a.annotate(f"Epoch = {(i + 1):d}", xy=(0.1, 0.1),
                                    xycoords='figure fraction')
        artists.append([container, epoch_index])

    ani = ArtistAnimation(fig=fig_a, artists=artists, interval=60)
    # ax[1].plot(history.history['cross_entropy'], label='cross entropy')
    # ax[1].set_title('cross entropy')
    # ax[1].legend()
    plt.show()
