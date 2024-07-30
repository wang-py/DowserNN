import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import utils
from training_visualization import weights_visualization_callback
from training_visualization import weights_history_visualizer
import tensorflow as tf
import matplotlib.pyplot as plt

# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)


def plot_model_accuracy(accuracy_values):
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
    plt.show()
    pass


def get_model_accuracy(model, X_validate, y_validate):
    y_predicted = model.predict(X_validate)
    assert y_validate.shape[0] == y_predicted.shape[0]
    y_validate = np.array(y_validate)
    y_predicted = np.array(y_predicted)
    accuracy_values = np.zeros(y_validate.shape[0])
    for i in range(accuracy_values.shape[0]):
        accuracy_values[i] = y_predicted[i].dot(y_validate[i].T)

    return accuracy_values


def get_low_accuracy_waters(accuracy_values):
    accuracy_threshold = 0.8
    water_index = np.where(accuracy_values < accuracy_threshold)[0]
    print(f"{water_index.shape[0]} waters have accuracy lower than" +
          f" {accuracy_threshold}")
    print("water indices:", water_index)
    np.savetxt('low_accuracy_water.txt', water_index, fmt='%d')


def plot_loss_history(history, test_loss):
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
    plt.show()


def build_NN(num_of_layers, N, input_dim, hidden_dim, learning_rate):
    model = Sequential()
    model.add(
        Dense(
            hidden_dim,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer='zeros'
        )
    )
    i = 1
    while (i < num_of_layers - 1):
        model.add(
            Dense(
                int(hidden_dim / (2 ** i)),
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer='zeros'
            )
        )
        i += 1
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=['accuracy'])
    model.build((N, input_dim))

    model.summary()
    return model


if __name__ == "__main__":
    # Generate training and validation data
    X_file = sys.argv[1]
    y_file = sys.argv[2]
    X_yes_file = X_file.split('.')[0] + '_yes.npy'
    y_yes_file = y_file.split('.')[0] + '_yes.npy'
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
    hidden_dim = 32
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])
    y_test = tf.convert_to_tensor(y[training_N:, :])
    # record weights during each training iteration
    # Create a neural network model
    num_of_layers = 2
    callback = weights_visualization_callback(num_of_layers)
    model = build_NN(num_of_layers, N, input_dim, hidden_dim,
                     learning_rate=0.001)
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
    plot_loss_history(history, test_loss)

    # plot confidence for water molecules
    accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    get_low_accuracy_waters(accuracy_values)
    plot_model_accuracy(np.sort(accuracy_values))

    # visualizing weights
    weights_history = callback.get_weights()
    weights_visualizer = weights_history_visualizer(weights_history)
    weights_visualizer.visualize()
    # weights_visualizer.save('layer_visualization.mp4')
