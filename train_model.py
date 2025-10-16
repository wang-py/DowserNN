import numpy as np
import random
import os
print('-' * 80)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import utils
from keras import saving
from training_visualization import weights_visualization_callback
from training_visualization import weights_history_visualizer
import tensorflow as tf

import timeit
# MONITOR devices used by Tensorflow
tf.config.set_visible_devices([], 'GPU')   # Set the device to CPU by hiding all GPU devices
import logging
try:
    # Set Python logging to INFO temporarily
    tf.get_logger().setLevel(logging.INFO)
    print('-' * 70)
    print("Available TensorFlow physical devices:")
    devices = tf.config.list_physical_devices()
    if not devices:
        print("No physical devices found. Check your TensorFlow installation.")
    else:
        for device in devices:
            print(f"- {device}")
finally:
    # Reset Python logging level to WARNING after printing
    tf.get_logger().setLevel(logging.WARNING)
    print("\nTensorFlow INFO logs have been suppressed for the rest of the script.")
    print('-' * 70)

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
parser.add_argument('-b', '--balance_y_no', type=float, default=1.0)
parser.add_argument('-r', '--restart', type=str)
parser.add_argument('-s', '--batch_size', type=int, default=32)
parser.add_argument('-e', '--epochs', type=int, default=100)

# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)

fig_count = 0        # Initializing figure count
def plt_savefig(fignm = None):
    if fignm is not None:
        plt.savefig(fignm, dpi = 200)
        return
    
    global training_pdb, fig_count
    fig_count += 1
    plt.savefig(f'{training_pdb}_nn{str(fig_count)}.png', dpi = 200)

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

def split_randomize_train_test_set(X_data, y_data, w_data, percent: float):
    """
    generates training and testing sets from all input data, the percentage of
    test data can be specified by "percent"
    ----------------------------------------------------------------------------
    X_data: ndarray
    all input X

    y_data: ndarray
    all input y

    w_data: ndarray
    all input sample_weights

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
    
    ## test_index = np.array(random.sample(indices, num_of_test_pts))
    ## train_index = np.array(list(set(indices) - set(test_index)))
    ## print(f'test_index[:30]: {test_index[:30]}')
    ## print(f'test_index[num_of_test_pts-30:]: {test_index[num_of_test_pts-30:]}')
    ## print(f'train_index[:30]: {train_index[:30]}')
    ## print(f'train_index[num_of_test_pts-30:]: {train_index[num_of_test_pts-30:]}')
    ## print(f'Num of Yes/No samples in the test_index: {np.sum(test_index < 2743)}(Yes), {np.sum(test_index >= 2743)}(No) ')
    ## print(f'Num of Yes/No samples in the train_index: {np.sum(train_index < 2743)}(Yes), {np.sum(train_index >= 2743)}(No) ')
    ## exit()
    test_X = tf.gather(X_data, indices=test_index)
    test_y = tf.gather(y_data, indices=test_index)
    test_w = tf.gather(w_data, indices=test_index)
    train_X = tf.gather(X_data, indices=train_index)
    train_y = tf.gather(y_data, indices=train_index)
    train_w = tf.gather(w_data, indices=train_index)

    return train_X, train_y, train_w, test_X, test_y, test_w

def plot_model_accuracy(accuracy_values, plot_title: str = 'model accuracy'):
    """
    function that plots the accuracy of water prediction
    ----------------------------------------------------------------------------
    accuracy_values: ndarray
    numpy array of accuracy values of water prediction

    plot_title: str
    title for the plot
    ----------------------------------------------------------------------------
    """
    accuracy_threshold = 0.5
    num_above_threshold = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold = float(num_above_threshold) / float(num_of_water)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(num_of_water), accuracy_values)
    ax.axhline(accuracy_threshold, color='k', linestyle='--',
               label=f'accuracy threshold = {accuracy_threshold}\n' +
               f'% predictions above threshold: {percent_above_threshold:.1%}')
    ax.set_xlabel("data index")
    ax.set_ylabel("confidence")
    ax.set_title(plot_title)
    ax.legend()
    plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
    plt.show()
    pass

def draw_model_accuracy_subset(model, X, y, setName: str, caseOpt: str = 'all'):
    """
    function that plots the accuracy of water prediction
    ----------------------------------------------------------------------------
    X: ndarray x 70
    numpy array of descriptors

    y: ndarray x 2
    numpy array of descriptors

    setName: str ('full', 'training', 'test', etc)

    caseOpt: str ('all', 'yes', 'no')
    ----------------------------------------------------------------------------
    """
    if caseOpt == 'all':
        subset_indices = range(X.shape[0])
        caseName = ''
    elif caseOpt == 'yes':
        subset_indices = np.where(y[:,0] == 1)[0]   # indices of Yes-cases
        caseName = 'Yes-cases'
    elif caseOpt == 'no':
        subset_indices = np.where(y[:,0] == 0)[0]   # indices of No-cases
        caseName = 'No-cases'
    else:
        print(f'ERROR in \"draw_model_accuracy_subset\":  caseOpt \"{caseOpt}\" is not supported.')
        exit()
    print(f'{setName} set {caseName}: nData = {len(subset_indices)}, indices[0:20]: {subset_indices[0:20]}')

    X_subset = tf.gather(X, indices=subset_indices)
    y_subset = tf.gather(y, indices=subset_indices)
    accuracies = get_model_accuracy(model, X_subset, y_subset)
    plot_model_accuracy(np.sort(accuracies), f'reproducing {setName} set {caseName}')
    return accuracies

def plot_dataset_prediction(model, X_data, y_data, plot_title: str = 'model accuracy'):
    """
    function that plots the accuracy of water prediction
    ----------------------------------------------------------------------------
    accuracy_values: ndarray
    numpy array of accuracy values of water prediction
    ----------------------------------------------------------------------------
    """
    accuracy_values = get_model_accuracy(model, X_data, y_data)

    accuracy_values = np.sort(accuracy_values) # Sort values acscending

    accuracy_threshold = 0.5
    num_above_threshold = np.sum(accuracy_values > accuracy_threshold)
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold = float(num_above_threshold) / float(num_of_water)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(num_of_water), accuracy_values)
    ax.axhline(accuracy_threshold, color='k', linestyle='--',
               label=f'accuracy threshold = {accuracy_threshold}\n' +
               f'% predictions above threshold: {percent_above_threshold:.1%}')
    plt.title(plot_title, fontsize=20, fontweight='bold')
    ax.set_xlabel("Index of data point", fontweight='bold')
    ax.set_ylabel("Confidence", fontweight='bold')
    ax.legend()
    plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
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
        #if i > -1 and i < 100: print(f"{i} predicted = {y_predicted[i]},  actual = {y_validate[i]}: accuracy_values = {accuracy_values[i]}")
    print(f"Validation Set has {len(accuracy_values)} data points")
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


def plot_loss_history(history, train_pdb, val_pdb, scale ='unscaled'):
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
    fig, ax = plt.subplots(figsize=(8, 6), layout='tight')
    training_loss = history.history['loss']
    nepochs = len(training_loss)
    epochs = range(1,nepochs+1)
    # check if there is validation loss
    try:
        validation_loss = history.history['val_loss']
    except KeyError:
        print('No test set used')
        validation_loss = None

    ax.plot(epochs, training_loss, 'b-', label=r'$\bf{training\ loss:}$ '
            + os.path.basename(train_pdb))
    if validation_loss:
        ax.plot(epochs, validation_loss, 'r-', label=r'$\bf{validation\ loss:}$ '
                + os.path.basename(val_pdb))
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    # ax.axhline(training_loss, color='b', linestyle='--',
    #           label='training cross entropy')
    # if test_loss is not None:
    #     ax.axhline(test_loss, color='r', linestyle='--',
    #                label='test cross entropy')
    ax.legend()
    global hidden_dim, batch_size, balance_y_no
    ax.set_title(r'$\bf{training\ and\ validation\ loss}$:' + f' nn({hidden_dim}), batch({batch_size}), balance({balance_y_no})', fontsize=12)

    if scale == 'unscaled':
        plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
        plt.show()
        return
    
    if not any(x in scale for x in ['xlog','ylog']):
        print(f'ERROR: unsupported scale parameter \"{scale}\" in the plot_loss_history() call.')
        print('Supported scale parameters are [\"unscaled\", \"*xlog*ylog*\", \"*xlog*\", \"*ylog*\"]')
        plt.show()
        return
    
    suff=''
    if 'xlog' in scale:
        ax.set_xscale('log')       # Set x-axis to logarithmic scale
        suff='xlog'
    if 'ylog' in scale:
        ax.set_yscale('log')       # Set y-axis to logarithmic scale
        if suff == 'xlog': suff = 'xylog'
        else:              suff = 'ylog'
    ax.set_title(r'$\bf{training\ and\ validation\ loss\ (log-scale)}$:' + f' nn({hidden_dim}), batch({batch_size}), balance({balance_y_no})', fontsize=12)
    global fig_count
    #fig_count += 1
    plt_savefig(f'{training_pdb}_nn{str(fig_count)}{suff}.png')   # Save figure with the figure count prefix "_nn{fig_count}"
    plt.show()

    ##  plt.show(block=False) # Show without blocking, so we can modify
    ##  user_input = plt.waitforbuttonpress()    # Wait for a button press or key press
    ##  # DO NOT CLOSE FIG WINDOW, just keyboard/mouse click!
    ##  
    ##  # Modify axis scale to logarithmic one
    ##  ax.set_title('training and validation loss (log-scale)')
    ##  ax.set_xscale('log')       # Set x-axis to logarithmic scale
    ##  ax.set_yscale('log')       # Set y-axis to logarithmic scale
    ##  
    ##  # fig.canvas.draw_idle()    # Redraw the figure to reflect the changes
    ##  global fig_count
    ##  plt_savefig(f'{training_pdb}_nn{str(fig_count)}log.png')   # Save figure with the figure count prefix "_nn{fig_count}"
    ##  plt.show()

def save_loss_history(history, train_pdb):
    """
    plots the training and validation loss
    ----------------------------------------------------------------------------
    history: history obj of training that contains the loss results

    train_pdb: pdb name of training structure
    ----------------------------------------------------------------------------
    Returns:
    None

    """
    # 1. Prepare the data
    epochs = np.arange(1, len(history.history['loss']) + 1)   # Create column of Epoch range(1,len(history))
    data = np.array(list(history.history.values())).T         # Transpose to have columns for each metric
    #data = np.column_stack(list(history.history.values()))

    # Combine epochs and data
    combined_data = np.hstack((epochs[:, np.newaxis], data))       # The first column will be epochs, followed by your metrics

    fmt_data = '%i'
    for key in history.history.keys():   # add format for all history metrics
        fmt_data = fmt_data + '\t%.5f' 

    header_string = "\t".join(history.history.keys())
    header_string = f'Epoch\t{header_string}'
    #print(f'header: {header_string} fmt_data={fmt_data}\ndata:\n{history.history.values()}')
    #print(f'data:\n{combined_data}')
    np.savetxt(train_pdb+'_nn.hist', combined_data, fmt=fmt_data, header=header_string, comments='#')
    

def plot_accuracy_history(history, train_pdb, val_pdb, scale ='unscaled'):
    """
    plots the training and validation accuracies
    ----------------------------------------------------------------------------
    history: history obj of training that contains the loss results

    train_pdb: pdb name of training structure

    val_pdb: pdb name of validation structure
    ----------------------------------------------------------------------------

    Returns:
    None

    """
    try:
        training_acc = history.history['accuracy']
    except KeyError:
        print('No accuracy metrics used')
        return
    # check if there is validation Accuracy
    try:
        validation_acc = history.history['val_accuracy']
    except KeyError:
        print('No validation set used')
        validation_acc = None
    # check if there is custom metrics: Accuracy for yes and no cases
    try:
        acc_p = history.history['acc_p']
        if validation_acc:
            val_acc_p = history.history['val_acc_p']
    except KeyError:
        print('No accuracy for yes-cases used.')
        acc_p = None
    try:
        acc_n = history.history['acc_n']
        if validation_acc:
            val_acc_n = history.history['val_acc_n']
    except KeyError:
        print('No accuracy for no-cases used.')
        acc_n = None
    n_subplots = 1
    if (acc_p or acc_n):
       n_subplots += 1
       if validation_acc:
            n_subplots += 1
    figsize_y = 1 + 3 * n_subplots
    nepochs = len(training_acc)
    epochs = range(1,nepochs+1)

    fig, axis = plt.subplots(nrows = n_subplots,ncols = 1, sharex='row', figsize=(8, figsize_y),layout='tight')   # layout='tight'
    #plt.subplots_adjust(left=0.09, bottom=0.08,top=0.94,right=0.98,wspace=1.0,hspace=0.1)   # Fine tuning but not suitable for log-scale
    global hidden_dim, batch_size, balance_y_no
    fig.suptitle(r'$\bf{Accuracy}$:' + f' nn({hidden_dim}), batch({batch_size}), balance({balance_y_no})', fontsize=14, y=0.99)
    fig.supxlabel('Epoch', fontsize=14, fontweight='bold',y=0.002)

    if n_subplots > 1: ax = axis[0]
    else:              ax = axis
    ax.plot(epochs,training_acc, 'b-', label=r'$\bf{training\ acc:}$ ' + os.path.basename(train_pdb))
    if validation_acc:
       ax.plot(epochs,validation_acc, 'r-', label=r'$\bf{validation\ acc:}$ ' + os.path.basename(val_pdb))
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

    ifig = 0
    if (acc_p or acc_n):
        ifig += 1
        axis[ifig].plot(epochs,training_acc, 'b-', label=r'$\bf{acc}$    : accuracy of all samples')
        if acc_p:
            axis[ifig].plot(epochs,acc_p, 'b:', label=r'$\bf{acc\_p}$: accuracy of positive samples')
        if acc_n:
            axis[ifig].plot(epochs,acc_n, 'b--', label=r'$\bf{acc\_n}$: accuracy of negative samples')
        axis[ifig].set_ylabel('Training', fontsize=12, fontweight='bold')
        if validation_acc:
            ifig += 1
            axis[ifig].plot(epochs,validation_acc, 'r-', label=r'$\bf{val\_acc}$    : accuracy of all samples')
            if acc_p:
                axis[ifig].plot(epochs,val_acc_p, 'r:', label=r'$\bf{val\_acc\_p}$: accuracy of positive samples')
            if acc_n:
                axis[ifig].plot(epochs,val_acc_n, 'r--', label=r'$\bf{val\_acc\_n}$: accuracy of negative samples')
            axis[ifig].set_ylabel('Validation', fontsize=12, fontweight='bold')
    for ifig in range(n_subplots):
        ax = axis
        if n_subplots > 1:
            ax = axis[ifig]
        ax.legend()
        ax.set_xlim(1, nepochs)
        if ifig < n_subplots - 1:
            axis[ifig].label_outer()     # Hide x-labels and tick labels for top suplots

    if scale == 'unscaled':
        plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
        plt.show()
        return
    
    if not any(x in scale for x in ['xlog','ylog']):
        print(f'ERROR: unsupported scale parameter \"{scale}\" in the plot_accuracy_history() call.')
        print('Supported scale parameters are [\"unscaled\", \"*xlog*ylog*\", \"*xlog*\", \"*ylog*\"]')
        plt.show()
        return
    
    suff=''
    for ifig in range(n_subplots):
        ax = axis
        if n_subplots > 1:
            ax = axis[ifig]
        if 'xlog' in scale:
            ax.set_xscale('log')       # Set x-axis to logarithmic scale
            suff='xlog'
        if 'ylog' in scale:
            ax.set_yscale('log')       # Set y-axis to logarithmic scale
            if suff == 'xlog': suff = 'xylog'
            else:              suff = 'ylog'
    fig.suptitle(r'$\bf{Accuracy\ (log-scale)}$:' + f' nn({hidden_dim}), batch({batch_size}), balance({balance_y_no})', fontsize=14, y=0.99)

    global fig_count
    #fig_count += 1
    plt_savefig(f'{training_pdb}_nn{str(fig_count)}{suff}.png')   # Save figure with the figure count prefix "_nn{fig_count}"
    plt.show()


class MinMaxNormalization(tf.keras.layers.Layer):
    # MinMaxNormalization Class that normalizdes along ALL () or a specific axis (axis=0)
    # Add Normalization Layer (aver-s.t.d.) https://www.architecture-performance.fr/ap_blog/saving-a-tf-keras-model-with-data-normalization/

    def __init__(self, **kwargs):
        super(MinMaxNormalization, self).__init__(**kwargs)
    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=0)
        max_val = tf.reduce_max(inputs, axis=0)
        return (inputs - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
class MinMaxNormalization2(tf.keras.layers.Layer):
    # MinMaxNormalization Class that normalizdes along ALL () or a specific axis (axis=0)
    def __init__(self, axis=None, **kwargs):
        super(MinMaxNormalization2, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super(MinMaxNormalization2, self).get_config()
        config.update({
            "axis": self.axis,
        })
        return config

    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=self.axis, keepdims=True)
        max_val = tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        return (inputs - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
def min_max_normalizing(data):
    # Reduce along columns (axis=0)
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    delta = max_val - min_val
    #print (f'min_val = {min_val}\nmax_val = {max_val}\nmax_val - min_val = {delta} ')
    #print (data[:2,:])
    # Avoid division by zero using np.where : norm_data = (data - min_val) / (max_val - min_val)
    norm_data = np.where(delta != 0, (data - min_val) / delta, 0.0)
    #print (norm_data[:2,:])
    return norm_data

##
## Implement custom metrics - individual accuracies for Yes- and No-cases:
## custom metrics examples: https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
##

# USE custom metrics definition via Class: acc_p(),acc_n() because the functions: 'acc_yes', 'acc_no' are 20% wrong at batch_size < 8.
# all three metrics=['Accuracy','Precision','Recall', 'BinaryAccuracy', f1_score] show the same values, only 'AUC' value is independent
# custom metrics examples: https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05

def f1_score(y_true, y_pred):
    # Round predictions to get binary values
    y_pred = tf.round(y_pred)
    
    # Cast to float for calculations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate true positives, false positives, false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1

def _numpy_print_array(array):
        print("Numpy array from custom metric:", array)
        return 0.0 # py_function requires a return value
def _numpy_print_values(array):
        print(f'num_positives:{array[0]} num_negatives:{array[1]}, true_positives:{array[2]}, true_negatives:{array[3]}')
        return 0.0 # py_function requires a return value

from keras import backend as K
def acc_yes(y_true, y_pred):
    # num_cases_arr      = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32), axis = 0)
    # y_pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    # # Calculate true positives: where y_true is 1 and y_pred_binary is 1
    # num_positives_arr  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), tf.float32), axis =0)
    # num_positives = num_cases_arr[0]
    # num_negatives = num_cases_arr[1]
    # true_positives=num_positives_arr[0]
    # true_negatives=num_positives_arr[1]
    #tf.print(y_pred[0:5,0])   # print prediction values (tensor length should be < batch_seze)
    num_positives  = K.sum(K.round(K.clip(y_true         , 0, 1)), axis = 0)[0]  # Get 0th-element of binary array[0 1]. Note arr[0]+arr[1]=Const=batch_size
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis = 0)[0]  # Note K.round(0.5) = 0.0 not 1.0
    accuracy_pos = true_positives / (num_positives + K.epsilon() )
    #tf.py_function(func = _numpy_print_values, inp=[[num_positives,num_negatives,true_positives,true_negatives]], Tout=tf.float32)
    #tf.print(f'num_positives:{num_positives} num_negatives:{num_negatives}, true_positives:{true_positives}, true_negatives:{true_negatives}')
    return accuracy_pos
def acc_no(y_true, y_pred):
    # y_pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    # num_negatives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32), axis = 0)[1]
    # true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), tf.float32), axis =0)[1]
    num_negatives  = K.sum(K.round(K.clip(y_true         , 0, 1)), axis = 0)[1]  # Get 1th-element of binary array[0 1]. Note arr[0]+arr[1]=Const=batch_size
    true_negatives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis = 0)[1]  # Note K.round(0.5) = 0.0 not 1.0
    accuracy_no = true_negatives / (num_negatives + K.epsilon() )
    return accuracy_no

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
        #num_cases_arr      = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32), axis = 0)
        #y_pred_rounded = tf.round(y_pred)
        #num_positives_arr  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_rounded, 1)), tf.float32), axis =0)
        #correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32)
        #self.total_correct.assign_add(tf.reduce_sum(correct_predictions))
        #self.total_samples.assign_add(tf.cast(tf.size(y_true), tf.float32))
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
    # Add Normalization Layer (aver-s.t.d.) https://www.architecture-performance.fr/ap_blog/saving-a-tf-keras-model-with-data-normalization/
    #model.add(tf.keras.layers.LayerNormalization()) # Add 141 trainable params
    #model.add(tf.keras.layers.experimental.preprocessing.Normalization())   # Add 141 Non-trainable params
    #norm_X = MinMaxNormalization(axis=0)  # SEACH "tensorflow normalization layer with min max value 2d array example"
    #model.add(MinMaxNormalization(axis=0)) # Add normalization layer
    #model.add(MinMaxNormalization()) # Add normalization layer
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
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'], weighted_metrics=[])
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'], weighted_metrics=[])
    #model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=['accuracy',acc_p(),acc_n()], weighted_metrics=[])   # weighted_metrics=['binary_crossentropy']
    # USE custom metrics definition via Class: acc_p(),acc_n() because the functions: 'acc_yes', 'acc_no' are 20% wrong at batch_size < 8.
    # all three metrics=['Accuracy','Precision','Recall', 'BinaryAccuracy', f1_score] show the same values, only 'AUC' value is independent
    # custom metrics examples: https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
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
    # NN model and training psarameters
    num_of_layers = 1
    hidden_dim = 4

    # Load training and validation data
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    balance_y_no = args.balance_y_no
    training_pdb = args.train_pdb
    testing_pdb = args.validate_pdb
    testing_percentage = args.test_percentage
    model_filename = args.output_filename
    X_file = training_pdb + "_CI_X.npy"
    y_file = training_pdb + "_CI_y.npy"
    X = np.load(X_file)
    y = np.load(y_file)
    #print(f"loaded y[0:10]:\n{y[0:10]}")
    #print(f'Last 2 descriptors: {X[-2:]}')

    # # NORMALIZE X columns
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # norm_X = scaler.fit_transform(X)
    # print (f'Original desc: {X[:2,:]}')
    # print (f'sklearn.scaler {norm_X[:2,:]}')
    # #norm_X = min_max_normalizing(X)
    # #print (f'my min_max_norm: {norm_X[:2,:]}')
    # #exit()
    # X_data = tf.convert_to_tensor(norm_X)

    # spliting data into training set and testing set
    X_data = tf.convert_to_tensor(X)
    y_data = tf.convert_to_tensor(y)
    input_dim = X_data.shape[1]
    N = X_data.shape[0]
    print(f'Loaded {N} descriptors of dimension {X_data.shape[1]}.')

    # Generate weights to balance under represented water data set during NN fitting
     #nYes = len(X_validate)
    nYes = np.sum(y[:,0] == 1)
    nNo  = np.sum(y[:,1] == 1)
    if nNo + nYes != N:
        print(f'ERROR: inconsistent y_data, number of Yes- and No-cases ({nYes}+{nNo}) is not equal to the total N = {N}.')
        exit()
    if nYes == 0 or nNo == 0:
        print(f'ERROR: number of Yes- or No-cases cannot be ZERO, nYes = {nYes}, nNo = {nNo}.')
        exit()
   
   # balance_y_no cases representation of water data in the loss function compare to No-cases, 1 means the same, 0.5/2 means twice under-/over-represented.
    weight_yes_multiplier = args.balance_y_no * float(nNo) / float(nYes)
    w_data = np.where(y[:, 0] == 1, weight_yes_multiplier, 1.0) # apply weight_yes_multiplier for Yes-cases(y[:, 0] == 1), otherwise weight = 1.0.
    # w_data = np.ones(N, dtype=float)
    # w_data[:nYes] = w_data[:nYes] * weight_yes_multiplier
    print(f'nYes = {nYes} nNo = {nNo} balance_y_no = {args.balance_y_no}: weight_yes_multiplier = {weight_yes_multiplier}')
    print(f'w_data[{nYes-3}:{nYes+3}] = {w_data[nYes-3:nYes+3]}')
    #print(f'y[{nYes-5}:{nYes+5}] = {y[nYes-5:nYes+5]}')

    # if not testing with another structure
    if testing_pdb is not None:
        X_train = X_data
        y_train = y_data
        X_file_test = testing_pdb + "_CI_X.npy"
        y_file_test = testing_pdb + "_CI_y.npy"
        X_test = tf.convert_to_tensor(np.load(X_file_test))
        y_test = tf.convert_to_tensor(np.load(y_file_test))
    else:
        testing_pdb = training_pdb
        if testing_percentage != 0:
            X_train, y_train, w_train, X_test, y_test, w_test =\
                split_randomize_train_test_set(X_data, y_data, w_data,
                                               percent=testing_percentage)
            # X_train, y_train, X_test, y_test =\
            #     generate_train_test_set(X_data, y_data,
            #                             percent=testing_percentage)
        else:
            X_train = X_data
            y_train = y_data
            X_test = None
            y_test = None

    #nYes_train = int (nYes * (1.0-testing_percentage))
    #nYes_test = int (nYes * testing_percentage)
    nYes_train = np.sum(y_train[:,0] == 1)
    nYes_test  = np.sum(y_test[:,0] == 1)
    print(f'nYes_train = {nYes_train} nYes_test = {nYes_test}')
    print(f'nTrain = {len(y_train)}, nWTrain = {len(w_train)}, nTest = {len(y_test)}:\nw_train[{nYes_train-10}:{nYes_train+10}] = {w_train[nYes_train-10:nYes_train+10]}')
    print(f'nTest = {len(X_test)}, nWTest = {len(w_test)}, nTest = {len(y_test)}:\nw_test[{nYes_test-10}:{nYes_test+10}] = {w_test[nYes_test-10:nYes_test+10]}')

    # record weights during each training iteration
    # Create a neural network model
    callback = weights_visualization_callback(num_of_layers)

    ##
    ## Initialize Model
    ##
    print('=' * 65)
    if args.restart is None:
        print(f"\nCreating a new model:")
        print(f"Layers={num_of_layers}, Layer_dim={hidden_dim}, TrainData_dim={len(y_train)}.\n")
        model = build_NN(num_of_layers, N, input_dim, hidden_dim, learning_rate=0.001)
    else:
        model_pth = args.restart
        print(f"\nRestarting training from the model:\n    \"{model_pth}\"\n")
        f = open(model_pth, 'r')
        f.close()
        try:
            model = tf.keras.models.load_model(model_pth)
            model.summary()
        except:
            print(f"Error: Cannot load the model {model_pth}.\nCheck keras-format compatibility.")
            exit()
    ##
    ## Train the model
    ##
    training_start_time = timeit.default_timer()
    if X_test is not None:
        history = model.fit(X_train, y_train, sample_weight = w_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test, w_test),
                            callbacks=callback, shuffle=True)   # by Default shuffle=True
    else:
        history = model.fit(X_train, y_train, sample_weight = w_train, epochs=epochs, batch_size=batch_size,
                            callbacks=callback, shuffle=True)   # by Default shuffle=True
    training_time = timeit.default_timer() - training_start_time
    print(f"NN training took {training_time:.2f} seconds")
    print('=' * 70)

    # save model
    if model_filename is None:
        model_filename = training_pdb + '.keras'
    save_model(model, model_filename)

    np.set_printoptions(precision=4, suppress=True)

    # plot training loss
    plot_loss_history(history, training_pdb, testing_pdb)
    plot_loss_history(history, training_pdb, testing_pdb, scale = 'xlog-ylog')    # Plot logarithmic scale  loss
    save_loss_history(history, training_pdb)   # save to file all history metrics
    # plot training Accuracies
    plot_accuracy_history(history, training_pdb, testing_pdb)
    plot_accuracy_history(history, training_pdb, testing_pdb, scale = 'xlog-ylog')

    # 1) plot training set accuracy
    # training_accuracies = get_model_accuracy(model, X_train, y_train)
    # plot_model_accuracy(np.sort(training_accuracies), 'reproducing training set')
    draw_model_accuracy_subset(model, X_train, y_train, 'training', 'all')   #  plot_model_accuracy for all dataset points
    draw_model_accuracy_subset(model, X_train, y_train, 'training', 'yes')   #  plot_model_accuracy for Yes-cases only
    draw_model_accuracy_subset(model, X_train, y_train, 'training',  'no')   #  plot_model_accuracy for No-cases only

    # 2) plot test set accuracy
    # test_accuracies = get_model_accuracy(model, X_test, y_test)
    # plot_model_accuracy(np.sort(test_accuracies), 'reproducing test set')
    draw_model_accuracy_subset(model, X_test, y_test, 'test', 'all')   #  plot_model_accuracy for all dataset points
    draw_model_accuracy_subset(model, X_test, y_test, 'test', 'yes')   #  plot_model_accuracy for Yes-cases only
    draw_model_accuracy_subset(model, X_test, y_test, 'test',  'no')   #  plot_model_accuracy for No-cases only

    # 3) plot full set accuracy
    accuracy_values = draw_model_accuracy_subset(model, X_data, y_data, 'full', 'yes')   #  plot_model_accuracy for Yes-cases only
    get_low_accuracy_waters(accuracy_values)                           #  Save low accuracy water indices
    draw_model_accuracy_subset(model, X_data, y_data, 'full',  'no')   #  plot_model_accuracy for No-cases only

    ## # 3) plot confidence for water molecules
    ## X_yes_file = X_file.rsplit('.', 1)[0] + '_yes.npy'
    ## y_yes_file = y_file.rsplit('.', 1)[0] + '_yes.npy'
    ## X_validate = tf.convert_to_tensor(np.load(X_yes_file))
    ## y_validate = tf.convert_to_tensor(np.load(y_yes_file))
    ## print(f'Loaded from yes-file: nYes = {X_validate.shape[0]}')
    ## accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    ## get_low_accuracy_waters(accuracy_values)
    ## plot_model_accuracy(np.sort(accuracy_values), 'reproducing water')
    ## # print(np.sort(accuracy_values)[0])
    ## 
    ## # 4) plot confidence for No-cases
    ## X_no_file = X_file.rsplit('.', 1)[0] + '_no.npy'
    ## y_no_file = y_file.rsplit('.', 1)[0] + '_no.npy'
    ## X_validate = tf.convert_to_tensor(np.load(X_no_file))
    ## y_validate = tf.convert_to_tensor(np.load(y_no_file))
    ## print(f'Loaded from no-file: nNo = {X_validate.shape[0]}')
    ## accuracy_values = get_model_accuracy(model, X_validate, y_validate)
    ## get_low_accuracy_waters(accuracy_values)
    ## plot_model_accuracy(np.sort(accuracy_values), 'reproducing no-cases')


    # visualizing weights
    weights_history = callback.get_weights()
    weights_visualizer = weights_history_visualizer(weights_history, mode='2d')
    weights_visualizer.visualize(interval=10, frametime=200)
    # weights_visualizer.save('layer_visualization_8OM1.mp4')
