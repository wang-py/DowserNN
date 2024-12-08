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


def decode_analysis(analysis_data, atom_types, res_types):
    """
    function that decodes the training data back to atom types and residue types
    ----------------------------------------------------------------------------
    analysis_data: N x 50 ndarray

    input data that contains information about water molecules and their environment
    ----------------------------------------------------------------------------
    Returns:
    analysis_data_decoded: N x 20 ndarray
    array that contains atom and residue names along with internal coordinates
    """
    analysis_data_decoded = np.zeros(analysis_data.shape[0], 20)
    list(atom_types.keys())[list(atom_types.values()).index(16)]

    return analysis_data_decoded


if __name__ == "__main__":
    analysis_data_arr = np.load(sys.argv[1])
    # load saved protein data
    atom_types, res_types = read_protein_data()
    analysis_data = decode_analysis(analysis_data_arr, atom_types, res_types)

    # components in input data
    # components = 70
    # pca = PCA(n_components=components)
    # X_scaled = StandardScaler().fit_transform(input_data_X)
    # pca_features = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    # ax[0].scatter(pca_features[0, :], pca_features[1, :])
    ax[0].hist(analysis_data[:, 0])
    ax[1].hist(analysis_data[:, 1])
    plt.show()

    pass
