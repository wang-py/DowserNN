import numpy as np
import sys
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
    X_file = sys.argv[1]
    y_file = sys.argv[2]
    X = np.load(X_file)
    y = np.load(y_file)

    training_N = int(X.shape[0] / 2)  # int(33000)
    X_data = tf.convert_to_tensor(X[:training_N, :])
    y_data = tf.convert_to_tensor(y[:training_N, :])
    input_dim = X_data.shape[1]
    hidden_dim = 20
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])
    y_test = tf.convert_to_tensor(y[training_N:, :])
    # record weights during each training iteration
    callback = weights_visualization_callback()
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
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss="binary_crossentropy", metrics=['accuracy'])
    model.build((N, input_dim))

    model.summary()
    epochs = 64
    # Train the model
    history = model.fit(X_data, y_data, epochs=epochs, batch_size=32,
                        callbacks=callback)
    y_predicted = model.predict(X_data)
    np.set_printoptions(precision=4, suppress=True)
    # print("expected output:\n", y_data)
    # print("predicted output:\n", y_predicted)

    # test with new data
    test_loss = None
    if training_N != X.shape[0]:
        test_loss, accuracy = model.evaluate(X_test, y_test)
        print(f"test loss: {test_loss}")  # , test accuracy: {accuracy:.2%}")
    # print("expected output:\n", y_test)
    # print("predicted output:\n", y_validate)
    # error_percent = np.sum(y_validate - y_test) / np.sum(y_test)

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='cross entropy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if test_loss is not None:
        ax.axhline(test_loss, color='k', linestyle='--',
                   label='test cross entropy')
    ax.set_title('Training Loss')
    ax.legend()

    # visualizing weights
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
