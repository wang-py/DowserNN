import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import utils
from keras import saving
import tensorflow as tf
import matplotlib.pyplot as plt

# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)


def plot_model_accuracy(accuracy_values):
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
    y_predicted = model.predict(X_validate)
    assert y_validate.shape[0] == y_predicted.shape[0]
    y_validate = np.array(y_validate)
    y_predicted = np.array(y_predicted)
    accuracy_values = np.zeros(y_validate.shape[0])
    for i in range(accuracy_values.shape[0]):
        accuracy_values[i] = y_predicted[i].dot(y_validate[i].T)

    return accuracy_values


def get_low_accuracy_waters(accuracy_values):
    accuracy_threshold = 0.5
    water_index = np.where(accuracy_values < accuracy_threshold)[0]
    print(f"{water_index.shape[0]} waters have accuracy lower than" +
          f" {accuracy_threshold}")
    entry = []
    for i in range(len(water_index)):
        entry.append(f"{water_index[i]} {accuracy_values[water_index[i]]}")
        # print(f"water indices: {water_index[i]} : {accuracy_values[water_index[i]]}")
    np.savetxt('low_accuracy_water.txt', np.array(entry), fmt='%s')


if __name__ == "__main__":
    # Generate training and validation data
    X_yes_file = sys.argv[1]
    y_yes_file = sys.argv[2]
    X_yes = np.load(X_yes_file)
    y_yes = np.load(y_yes_file)

    X_validate = tf.convert_to_tensor(X_yes)
    y_validate = tf.convert_to_tensor(y_yes)

    # record weights during each training iteration
    # Create a neural network model
    try:
        model = saving.load_model('test_data/DowserNN.keras')
    except ValueError:
        print("No exising model found")
        exit()
    np.set_printoptions(precision=4, suppress=True)
    # print("expected output:\n", y_data)
    # print("predicted output:\n", y_predicted)

    # test with new data
    test_loss, accuracy = model.evaluate(X_validate, y_validate)
    print(f"test loss: {test_loss}")  # , test accuracy: {accuracy:.2%}")
    # print("expected output:\n", y_test)
    # print("predicted output:\n", y_validate)
    # error_percent = np.sum(y_validate - y_test) / np.sum(y_test)

    # plot training loss

    # plot confidence for water molecules
    accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    get_low_accuracy_waters(accuracy_values)
    plot_model_accuracy(np.sort(accuracy_values))
    # print(np.sort(accuracy_values)[0])
