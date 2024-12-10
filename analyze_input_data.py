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

    return atom_types, res_types


atom_types, res_types = read_protein_data()


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
    atom_data_str = np.zeros(data_pts, 10)
    res_data_str = np.zeros(data_pts, 10)

    for i in range(data_pts):
        atom_data_str[i] = list(atom_types.keys())[list(atom_types.values()).index(atom_data[i])]
        res_data_str[i] = list(atom_types.keys())[list(atom_types.values()).index(atom_data[i])]

    # list(atom_types.keys())[list(atom_types.values()).index(16)]

    return atom_data_str, res_data_str


def plot_atom_res_dist(atom_data, res_data):
    fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(pca_features[0, :], pca_features[1, :])
    ax[0].hist(atom_data[:, 0])
    ax[1].hist(res_data[:, 0])
    plt.show()
    pass


if __name__ == "__main__":
    analysis_data_arr = np.load(sys.argv[1])
    # load saved protein data
    atom_data, res_data = get_atom_res_data(analysis_data_arr)
    atom_data_str, res_data_str = decode_atom_res_data(atom_data, res_data)
    plot_atom_res_dist(atom_data, res_data)

    # components in input data
    # components = 70
    # pca = PCA(n_components=components)
    # X_scaled = StandardScaler().fit_transform(input_data_X)
    # pca_features = pca.fit_transform(X_scaled)

    pass
