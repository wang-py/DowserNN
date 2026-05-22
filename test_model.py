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
parser.add_argument('-k', '--sort_key', type=str)


# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)

fig_count = 0        # Initializing figure count
def plt_savefig():
    global args, fig_count
    fig_count += 1
    plt.savefig(f'{args.test_file}_test{str(fig_count)}.png', dpi = 200)

def plot_model_accuracy(accuracy_values, figtitle=None, sorted_val=True):
    if sorted_val:
        accuracy_values = np.sort(accuracy_values)
    accuracy_threshold = 0.5
    num_above_threshold = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold = num_above_threshold / num_of_water
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(figtitle)
    ax.bar(np.arange(num_of_water), accuracy_values)
    ax.axhline(accuracy_threshold, color='k', linestyle='--',
               label=f'accuracy threshold = {accuracy_threshold}\n' +
               f'% water above threshold: {percent_above_threshold:.1%}')
    ax.set_xlabel("index")
    ax.set_ylabel("confidence")
    ax.legend()
    plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
    plt.show()
    pass


def gaussian(energies, cutoff=-4):
    return np.exp(-(cutoff - energies) ** 2)


def plot_water_data(acc_and_bfactors, sorted_by='accuracy'):
    accuracy_threshold = 0.5
    energy_threshold = -4.0
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    print(acc_and_bfactors)
    if sorted_by == 'accuracy':
        acc_and_bfactors = acc_and_bfactors[ acc_and_bfactors[:, 0].argsort()]
        P = gaussian(acc_and_bfactors[:, 0], cutoff=accuracy_threshold)
        P_threshold = 0.5
    elif sorted_by == 'energy':
        acc_and_bfactors = acc_and_bfactors[ acc_and_bfactors[:, 1].argsort() ]
        P = gaussian(acc_and_bfactors[:, 1], cutoff=energy_threshold)
        P_threshold = 0.5


    accuracy_values = acc_and_bfactors[:, 0]
    water_energies = acc_and_bfactors[:, 1]
    num_above_threshold_acc = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold_acc = num_above_threshold_acc / num_of_water
    num_below_threshold_E = np.sum(water_energies < energy_threshold)
    percent_above_threshold_E = num_below_threshold_E / num_of_water
    ax[0].bar(np.arange(num_of_water), accuracy_values)
    ax[0].axhline(accuracy_threshold, color='k', linestyle='--',
                  label=f'accuracy threshold = {accuracy_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_acc:.0%}')
    ax[0].set_ylabel("confidence")
    ax[0].legend()
    ax[1].bar(np.arange(num_of_water), water_energies)
    ax[1].axhline(energy_threshold, color='k', linestyle='--',
                  label=f'energy threshold = {energy_threshold} kcal/mol\n' +
                  f'% water below threshold: {percent_above_threshold_E:.0%}')
    # ax[0].set_xlabel("water index")
    ax[1].set_ylabel("energy [kcal/mol]")
    ax[1].legend()
    # ax[0].set_xlabel("water index")

    num_above_threshold_P = np.sum(P > P_threshold)
    percent_above_threshold_P = num_above_threshold_P / num_of_water
    ax[2].bar(np.arange(num_of_water), P)
    ax[2].axhline(P_threshold, color='k', linestyle='--',
                  label=f'probability threshold = {P_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_P:.0%}')
    ax[2].set_ylabel("probability")
    ax[2].legend()
    plt.xlabel("water index")
    plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
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
        dowser_energies = [float(x[60:67]) for x in data]
    return np.array(dowser_energies)

def get_bfactors(pdb):
    with open(pdb, 'r') as records:
        atoms = records.readlines()
        # PDB Format: https://cupnet.net/pdb-format/
        bfactors = [float(a[60:67]) for a in atoms if ( (a[17:20] =='HOH') and (a[0:4] =='ATOM') )]  # Take values for (resnm HOH and ATOM) records only.
        # bfactors = []
        # for a in atoms:
        #     try:
        #        bfac = float(a[60:67])
        #        bfactors.append(bfac)
        #     except ValueError:
        #        print (f'rec:{a}')
        #        print(f'ERROR: the B-factor field (61:68) in PDB cannot be converted to float')
        #        exit()
    return np.array(bfactors)

def get_records_bfac(pdb):
    with open(pdb, 'r') as atoms:
        records = atoms.readlines()
        bfactors = [float(rec[60:67]) for rec  in records]
    return np.array(records),np.array(bfactors)

def savepdb_new_bfac(pdb,values,val_suff):
    # SAVE PDB with accuracies in place of B-factors rec[60:67]
    import os
    with open(pdb, 'r') as atoms:
        records = atoms.readlines()
    nrec = len(records)
    nrec_HOH = sum(1 for rec in records if ( (rec[17:20] =='HOH') and (rec[0:4] =='ATOM')) )
    if (nrec_HOH != len(values)):
        print(f"Error in replace_bfac: the number of new b-factor values ({len(values)}) differs from number of HOH records ({nrec_HOH}) in {pdb}\nExit")
        exit()
    abs_path = os.path.abspath(args.water_pdb)
    path_wo_ext, _ = os.path.splitext(abs_path) # Split the path into root and extension
    new_pdb = f'{path_wo_ext}_{val_suff}.pdb'
    try:
        f = open(new_pdb, 'w')
    except OSError:
        print(f"Error: cannot open file for writing {new_pdb}\nExit")
        exit()

    # REPLACE B-FAC ONLY FOR HOH ATOMS
    iw = 0
    for i in range(nrec):
        if ( (records[i][17:20] =='HOH') and (records[i][0:4] =='ATOM') ):
            records[i] = "{}{:6.2f}{}".format(records[i][:60], values[iw], records[i][67:])  # PDB format https://cupnet.net/pdb-format/
            iw += 1
    f.writelines(records)                    # Read lines with "\n" at the end
    f.close()

import tensorflow.keras.backend as K
class acc_p(tf.keras.metrics.Metric):
    def __init__(self, name='acc_p', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='tc', initializer='zeros')
        self.total_samples = self.add_weight(name='ts', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        num_samples   = K.sum(K.round(K.clip(y_true         , 0, 1)), axis = 0)[0]
        num_predicted = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis = 0)[0]
        self.total_correct.assign_add(num_predicted)
        self.total_samples.assign_add(num_samples)
    def result(self):
        return self.total_correct / self.total_samples
    def reset_state(self):
        self.total_correct.assign(0.)
        self.total_samples.assign(0.)

class acc_n(tf.keras.metrics.Metric):
    def __init__(self, name='acc_n', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='tc', initializer='zeros')
        self.total_samples = self.add_weight(name='ts', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        num_samples   = K.sum(K.round(K.clip(y_true         , 0, 1)), axis = 0)[1]
        num_predicted = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis = 0)[1]
        self.total_correct.assign_add(num_predicted)
        self.total_samples.assign_add(num_samples)
    def result(self):
        return self.total_correct / self.total_samples
    def reset_state(self):
        self.total_correct.assign(0.)
        self.total_samples.assign(0.)

if __name__ == "__main__":
    # Generate training and validation data
    args = parser.parse_args()
    if args.sort_key is None:
        args.sort_key = 'accuracy'
    X_yes_file_suffix = "_CI_X_yes.npy"
    y_yes_file_suffix = "_CI_y_yes.npy"
    X_yes_file = args.test_file + X_yes_file_suffix
    y_yes_file = args.test_file + y_yes_file_suffix
    X_yes = np.load(X_yes_file)
    y_yes = np.load(y_yes_file)
    X_validate_yes = tf.convert_to_tensor(X_yes)
    y_validate_yes = tf.convert_to_tensor(y_yes)

    try:
        #model = saving.load_model(args.model)
        f = open(args.model, 'r')
        f.close()
        from keras.models import load_model
        model = load_model(args.model, {'acc_p': acc_p, 'acc_n': acc_n})
        model.summary()
    except ValueError:
        print("No exising model found")
        exit()
    np.set_printoptions(precision=4, suppress=True)

    accuracy_values_yes = get_model_accuracy(model,
                                             X_validate_yes, y_validate_yes)
    # test with new data
    # test with new data
    loss, accuracy, acc_p, acc_n = model.evaluate(X_validate_yes, y_validate_yes)
    print(f"loss: {loss:.4f}")  # , test accuracy: {accuracy:.2%}")
    print(f"accuracy: {accuracy:.2%}")  # , test accuracy: {accuracy:.2%}")
    print(f"acc_p: {acc_p:.2%}")  # , test accuracy: {accuracy:.2%}")
    #print(f"acc_n: {acc_n:.2%}")  # , test accuracy: {accuracy:.2%}")

    if args.water_pdb:
        bfactors = get_bfactors(args.water_pdb)
        print(f'n_bfac = {len(bfactors)}, n_acc = {len(accuracy_values_yes)}')
        acc_and_bfactors = np.c_[accuracy_values_yes, bfactors]
        plot_water_data(acc_and_bfactors, sorted_by=args.sort_key)
        savepdb_new_bfac(args.water_pdb,accuracy_values_yes * 100,'acc')  # Save pdb with accuracies in position of b-factors

    # plot confidence for water molecules
    # get_low_accuracy_waters(accuracy_values_yes)
    plot_model_accuracy(accuracy_values_yes, figtitle='tested with yes cases')




    # 2) Test no-cases
    X_no_file_suffix = "_CI_X_no.npy"
    y_no_file_suffix = "_CI_y_no.npy"
    X_no_file = args.test_file + X_no_file_suffix
    y_no_file = args.test_file + y_no_file_suffix
    X_no = np.load(X_no_file)
    y_no = np.load(y_no_file)
    X_validate_no = tf.convert_to_tensor(X_no)
    y_validate_no = tf.convert_to_tensor(y_no)
    # flip the accuracy to reflect water prediction result
    #accuracy_values_no = 1 - get_model_accuracy(model, X_validate_no, y_validate_no)
    accuracy_values_no = get_model_accuracy(model, X_validate_no, y_validate_no)
    plot_model_accuracy(accuracy_values_no, figtitle='tested with no cases')

