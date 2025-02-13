import numpy as np
import random
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import utils
from keras import saving
from training_visualization import weights_visualization_callback
from training_visualization import weights_history_visualizer
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
        prog='train_model.py',
        description='script that trains neural network model and\
                saves it to file',
        )
parser.add_argument('-t', '--train_pdb', type=str)
parser.add_argument('-p', '--test_percentage', type=float, default=0.2)
parser.add_argument('-v', '--validate_pdb', type=str)
parser.add_argument('-o', '--output_filename', type=str)


# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)


def generate_train_test_set(X_data, y_data, percent: float):
    """
    generates training and testing sets from all input data, the percentage of
    test data can be specified by "percent"
    ----------------------------------------------------------------------------
    X_data: ndarray
    all input X

    y_data: ndarray
    all input y

    percent: float
    percentage of testing data in all data
    ----------------------------------------------------------------------------
    Returns:
    test_X: ndarray
    X data for testing

    test_y
    y data for testing
    """
    index_range = X_data.shape[0]
    indices = range(index_range)
    num_of_test_pts = int(index_range * percent)
    test_index = random.sample(indices, num_of_test_pts)
    train_index = list(set(indices) - set(test_index))
    test_X = tf.gather(X_data, indices=test_index)
    test_y = tf.gather(y_data, indices=test_index)
    train_X = tf.gather(X_data, indices=train_index)
    train_y = tf.gather(y_data, indices=train_index)

    return train_X, train_y, test_X, test_y


def plot_model_accuracy(accuracy_values):
    """
    function that plots the accuracy of water prediction
    ----------------------------------------------------------------------------
    accuracy_values: ndarray
    numpy array of accuracy values of water prediction
    ----------------------------------------------------------------------------
    """
    accuracy_threshold = 0.5
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
    """
    function that evaluates the accuracy of water prediction
    ----------------------------------------------------------------------------
    model: obj
    pre-trained model used for evaluation

    X_validate: ndarray
    descriptors of water molecules

    y_validate: ndarray
    yes/no results for water molecules
    ----------------------------------------------------------------------------

    Returns:
    accuracy_values: ndarray
    accuracy values of predicted water molecules
    """
    y_predicted = model.predict(X_validate)
    assert y_validate.shape[0] == y_predicted.shape[0]
    y_validate = np.array(y_validate)
    y_predicted = np.array(y_predicted)
    accuracy_values = np.zeros(y_validate.shape[0])
    for i in range(accuracy_values.shape[0]):
        accuracy_values[i] = y_predicted[i].dot(y_validate[i].T)

    return accuracy_values


def get_low_accuracy_waters(accuracy_values):
    """
    finds the index of water with a accuracy lower than 50%
    ----------------------------------------------------------------------------
    accuracy_values: ndarray
    accuracy values of predicted water molecules
    ----------------------------------------------------------------------------

    Returns:
    saves the index and accuracy values of water with accuracy lower than 50%
    to a txt file named "low_accuracy_water.txt"

    """
    accuracy_threshold = 0.5
    water_index = np.where(accuracy_values < accuracy_threshold)[0]
    print(f"{water_index.shape[0]} waters have accuracy lower than" +
          f" {accuracy_threshold}")
    entry = []
    for i in range(len(water_index)):
        entry.append(f"{water_index[i]} {accuracy_values[water_index[i]]}")
        # print(f"water indices: {water_index[i]} : {accuracy_values[water_index[i]]}")
    np.savetxt('low_accuracy_water.txt', np.array(entry), fmt='%s')


def plot_loss_history(history, train_pdb, val_pdb):
    """
    plots the training and validation loss
    ----------------------------------------------------------------------------
    history: history obj of training that contains the loss results

    train_pdb: pdb name of training structure

    val_pdb: pdb name of validation structure
    ----------------------------------------------------------------------------

    Returns:
    None

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    ax.plot(training_loss, 'b-', label='training loss '
            + os.path.basename(train_pdb))
    ax.plot(validation_loss, 'r-', label='validation loss '
            + os.path.basename(val_pdb))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # ax.axhline(training_loss, color='b', linestyle='--',
    #           label='training cross entropy')
    # if test_loss is not None:
    #     ax.axhline(test_loss, color='r', linestyle='--',
    #                label='test cross entropy')
    ax.set_title('training and validation loss')
    ax.legend()
    plt.show()


def build_NN(num_of_layers: int, N: int, input_dim: int, hidden_dim: int,
             learning_rate: float):
    """
    function that builds the neural network
    ----------------------------------------------------------------------------
    num_of_layers: int
    number of layers of the neural network

    N: int
    size of one descriptor

    input_dim: int
    number of descriptors

    hidden_dim: int
    dimension of the hidden layers

    learning_rate: float
    learning rate of back propagation
    ----------------------------------------------------------------------------

    Returns:
    model: obj
    Neural network object

    """
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


def save_model(model, output_filename: str):
    """
    saves trained model to file
    ----------------------------------------------------------------------------
    model: obj
    neural network object

    output_filename: str
    filename for the model
    ----------------------------------------------------------------------------
    """
    model.save(output_filename)


if __name__ == "__main__":
    # Generate training and validation data
    args = parser.parse_args()
    training_pdb = args.train_pdb
    testing_pdb = args.validate_pdb
    testing_percentage = args.test_percentage
    model_filename = args.output_filename
    X_file = training_pdb + "_CI_X.npy"
    y_file = training_pdb + "_CI_y.npy"
    X = np.load(X_file)
    y = np.load(y_file)
    # spliting data into training set and testing set
    X_data = tf.convert_to_tensor(X)
    y_data = tf.convert_to_tensor(y)
    input_dim = X_data.shape[1]
    hidden_dim = 64
    N = X_data.shape[0]
    # if not testing with another structure
    if testing_pdb is not None:
        X_train = X_data
        y_train = y_data
        X_file_test = testing_pdb + "_CI_X.npy"
        y_file_test = testing_pdb + "_CI_y.npy"
        X_test = tf.convert_to_tensor(np.load(X_file_test))
        y_test = tf.convert_to_tensor(np.load(y_file_test))
    else:
        X_train, y_train, X_test, y_test =\
            generate_train_test_set(X_data, y_data,
                                    percent=testing_percentage)
        testing_pdb = training_pdb
    X_yes_file = X_file.split('.')[0] + '_yes.npy'
    y_yes_file = y_file.split('.')[0] + '_yes.npy'
    X_validate = tf.convert_to_tensor(np.load(X_yes_file))
    y_validate = tf.convert_to_tensor(np.load(y_yes_file))

    # record weights during each training iteration
    # Create a neural network model
    num_of_layers = 3
    callback = weights_visualization_callback(num_of_layers)
    try:
        model = saving.load_model('test_data/DowserNN.keras')
        model.summary()
    except ValueError:
        print("No exising model found, creating a new model")
        model = build_NN(num_of_layers, N, input_dim, hidden_dim,
                         learning_rate=0.0005)
    epochs = 30
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=callback)
    np.set_printoptions(precision=4, suppress=True)

    # plot training loss
    plot_loss_history(history, training_pdb, testing_pdb)

    # plot confidence for water molecules
    accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    get_low_accuracy_waters(accuracy_values)
    plot_model_accuracy(np.sort(accuracy_values))
    # print(np.sort(accuracy_values)[0])

    # visualizing weights
    weights_history = callback.get_weights()
    weights_visualizer = weights_history_visualizer(weights_history, mode='2d')
    weights_visualizer.visualize(interval=10, frametime=200)
    # weights_visualizer.save('layer_visualization_8OM1.mp4')
    # save model
    if model_filename is None:
        model_filename = training_pdb + '.keras'
    save_model(model, model_filename)
