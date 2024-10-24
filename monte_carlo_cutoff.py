import numpy as np
import matplotlib.pyplot as plt
import sys


def gaussian(energy, cutoff=-4):
    return np.exp(-(cutoff - energy) ** 2)


def monte_carlo_test(water_energies):
    num_of_water = water_energies.shape[0]
    water_category = np.zeros(num_of_water)

    for i in range(num_of_water):
        P = gaussian(water_energies[i])
        R = np.random.uniform()

        if R < P:
            water_category[i] = True
        else:
            water_category[i] = False

    return water_category


def get_water_probability(num_of_trials, water_energies):
    water_trials = np.zeros([num_of_trials, water_energies.shape[0]])

    for i in range(num_of_trials):
        water_trials[i] = monte_carlo_test(water_energies)

    water_probabilities = np.mean(water_trials, axis=0)

    return water_probabilities


def get_water_energies_from_pdb(water_pdb):
    with open(water_pdb, 'r') as w:
        data = w.readlines()
        water_energies = np.array([float(line[60:66]) for line in data
                                   if 'OW' in line])
    return water_energies


def plot_energy_and_probabilities(water_energies, water_probabilities):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    water_index = np.arange(water_energies.shape[0]) + 1
    ax[0].plot(water_index, water_energies, '--o')
    ax[0].set_title("water energies")
    ax[0].set_ylabel("kCal")
    ax[0].set_xlabel("water index")
    ax[0].axhline(-4, label="Dowser cutoff = -4 kCal/mol")
    ax[0].legend()

    ax[1].plot(water_index, water_probabilities, '--o')
    ax[1].set_title("water probabilities of being YES")
    ax[1].set_ylabel("probability")
    ax[1].set_xlabel("water index")

    label_size = 14
    for i, txt in enumerate(water_index):
        ax[0].annotate(txt, (water_index[i], water_energies[i]),
                       fontsize=label_size)
        ax[1].annotate(txt, (water_index[i], water_probabilities[i]),
                       fontsize=label_size)
    plt.show()
    pass


if __name__ == '__main__':
    input_pdb = sys.argv[1]
    num_of_trials = 100
    water_energies = get_water_energies_from_pdb(input_pdb)
    water_probabilities = get_water_probability(num_of_trials, water_energies)
    print("the energies of water are:\n", water_energies)
    print("the probabilities of yes:\n", water_probabilities)

    plot_energy_and_probabilities(water_energies, water_probabilities)

    pass
