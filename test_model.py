import numpy as np
from keras import utils
from keras import saving
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
        prog='test_model.py',
        description='script that tests neural network model prediction\
                accuracy',
        )
parser.add_argument('-t', '--test_file', type=str)
parser.add_argument('-w', '--water_pdb', type=str)
parser.add_argument('-m', '--model', type=str)


# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)


def plot_model_accuracy(accuracy_values, sorted_val=True):
    if sorted_val:
        accuracy_values = np.sort(accuracy_values)
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


def gaussian(energies, cutoff=-4):
    return np.exp(-(cutoff - energies) ** 2)


def plot_water_data(acc_and_energies, sorted_by='accuracy'):
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    print(acc_and_energies)
    if sorted_by == 'accuracy':
        acc_and_energies = acc_and_energies[
            acc_and_energies[:, 0].argsort()
            ]
    elif sorted_by == 'energy':
        acc_and_energies = acc_and_energies[
            acc_and_energies[:, 1].argsort()
            ]

    accuracy_values = acc_and_energies[:, 0]
    water_energies = acc_and_energies[:, 1]
    accuracy_threshold = 0.5
    num_above_threshold_acc = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold_acc = num_above_threshold_acc / num_of_water
    energy_threshold = -4
    num_above_threshold_E = np.sum(water_energies > energy_threshold)
    percent_above_threshold_E = num_above_threshold_E / num_of_water
    ax[0].bar(np.arange(num_of_water), accuracy_values)
    ax[0].axhline(accuracy_threshold, color='k', linestyle='--',
                  label=f'accuracy threshold = {accuracy_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_acc:.0%}')
    ax[0].set_ylabel("confidence")
    ax[0].legend()
    ax[1].bar(np.arange(num_of_water), water_energies)
    ax[1].axhline(energy_threshold, color='k', linestyle='--',
                  label=f'energy threshold = {energy_threshold} kCal/mol\n' +
                  f'% water above threshold: {percent_above_threshold_E:.0%}')
    # ax[0].set_xlabel("water index")
    ax[1].set_ylabel("energy [kCal/mol]")
    ax[1].legend()
    # ax[0].set_xlabel("water index")
    P = gaussian(water_energies, cutoff=energy_threshold)
    P_threshold = 0.5
    num_above_threshold_P = np.sum(P > P_threshold)
    percent_above_threshold_P = num_above_threshold_P / num_of_water
    ax[2].bar(np.arange(num_of_water), P)
    ax[2].axhline(P_threshold, color='k', linestyle='--',
                  label=f'probability threshold = {P_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_P:.0%}')
    ax[2].set_ylabel("probability")
    ax[2].legend()
    plt.xlabel("water index")
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


def get_dowser_energies(water_pdb):
    with open(water_pdb, 'r') as water:
        data = water.readlines()
        dowser_energies = [float(x[61:67]) for x in data]

    return np.array(dowser_energies)


if __name__ == "__main__":
    # Generate training and validation data
    args = parser.parse_args()
    X_yes_file_suffix = "_CI_X_yes.npy"
    y_yes_file_suffix = "_CI_y_yes.npy"
    X_yes_file = args.test_file + X_yes_file_suffix
    y_yes_file = args.test_file + y_yes_file_suffix

    X_yes = np.load(X_yes_file)
    y_yes = np.load(y_yes_file)

    X_validate = tf.convert_to_tensor(X_yes)
    y_validate = tf.convert_to_tensor(y_yes)

    try:
        model = saving.load_model(args.model)
    except ValueError:
        print("No exising model found")
        exit()
    np.set_printoptions(precision=4, suppress=True)

    if args.water_pdb:
        dowser_energies = get_dowser_energies(args.water_pdb)

    # test with new data
    test_loss, accuracy = model.evaluate(X_validate, y_validate)
    print(f"test loss: {test_loss}")  # , test accuracy: {accuracy:.2%}")

    # plot confidence for water molecules
    accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    acc_and_energies = np.c_[accuracy_values, dowser_energies]
    get_low_accuracy_waters(accuracy_values)
    plot_model_accuracy(accuracy_values)
    plot_water_data(acc_and_energies, sorted_by='accuracy')
