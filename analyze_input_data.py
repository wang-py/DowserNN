import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler


def read_protein_data(protein_data_dir='protein_data/'):
    with open(protein_data_dir + 'atom_types.pkl', 'rb') as a:
        atom_types = pickle.load(a)
    with open(protein_data_dir + 'residue_types.pkl', 'rb') as r:
        res_types = pickle.load(r)

    atom_types_decoded = dict((v, k) for k, v in atom_types.items())
    res_types_decoded = dict((v, k) for k, v in res_types.items())

    return atom_types_decoded, res_types_decoded


atom_types_decoded, res_types_decoded = read_protein_data()


def get_atom_res_data(analysis_data):
    """
    function that extracts atom codes and residue codes from analysis data
    ----------------------------------------------------------------------------
    analysis_data: N x 50 ndarray

    input data that contains information about water molecules and their
    environment
    ----------------------------------------------------------------------------
    Returns:
    atom_data: N x 10 ndarray
    res_data: N x 10 ndarray
    arrays that contain atom and residue codes
    """
    atom_data = analysis_data[:, 0::5]
    res_data = analysis_data[:, 1::5]

    return atom_data, res_data


def decode_atom_res_data(atom_data, res_data):
    """
    function that decodes the training data back to atom types and residue
    types
    ----------------------------------------------------------------------------
    analysis_data: N x 50 ndarray

    input data that contains information about water molecules and their
    environment
    ----------------------------------------------------------------------------
    Returns:
    atom_data: N x 10 ndarray
    res_data: N x 10 ndarray
    arrays that contain atom and residue codes
    """
    data_pts = atom_data.shape[0]
    atom_data_str = []
    res_data_str = []

    for i in range(data_pts):
        atom_data_str.append([atom_types_decoded[a_i] for a_i in atom_data[i]])
        res_data_str.append([res_types_decoded[r_i] for r_i in res_data[i]])

    # list(atom_types.keys())[list(atom_types.values()).index(16)]

    return np.array(atom_data_str), np.array(res_data_str)


def plot_atom_res_dist(atom_data_str, res_data_str):
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))
    num_of_bins_atom = len(atom_types_decoded.values())
    num_of_bins_res = len(res_types_decoded.values())
    ax[0].hist(atom_data_str[:, 0], bins=num_of_bins_atom, edgecolor='black')
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, fontsize=6)
    ax[1].hist(res_data_str[:, 0], bins=num_of_bins_res, edgecolor='black')
    plt.show()
    pass


if __name__ == "__main__":
    analysis_data_arr = np.load(sys.argv[1])
    # load saved protein data
    atom_data, res_data = get_atom_res_data(analysis_data_arr)
    atom_data_str, res_data_str = decode_atom_res_data(atom_data, res_data)
    plot_atom_res_dist(atom_data_str, res_data_str)

    # components in input data
    # components = 70
    # pca = PCA(n_components=components)
    # X_scaled = StandardScaler().fit_transform(input_data_X)
    # pca_features = pca.fit_transform(X_scaled)

    pass
