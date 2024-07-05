import sys
# import os
import numpy as np

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21}  # , 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'S': 4, 'H': 5}  # , 'Others': 6}


def get_internal_coords(relative_coors):
    """
    calculate internal coordinates based on relative vectors
    ----------------------------------------------------------------------------
    relative_coors: ndarray N x 3
    Array of relative coordinates from water to any other atom
    ----------------------------------------------------------------------------
    Returns:
    internal_coords: ndarray N x 3
    internal coordinates based on relative positions
    """
    N = relative_coors.shape[0]
    internal_coords = np.zeros([N, 3])
    r1 = relative_coors[0, :]
    r2 = relative_coors[1, :]
    internal_coords[0] = np.array([r1.dot(r1.T), 0, 0])
    internal_coords[1] = np.array([r2.dot(r2.T), r2.dot(r1.T), 0])

    for i in range(2, N):
        r_a = relative_coors[i]
        r_b = relative_coors[i - 1]
        r_c = relative_coors[i - 2]
        internal_coords[i] = np.array([r_a.dot(r_a.T),
                                       r_a.dot(r_b.T),
                                       r_a.dot(r_c.T)])

    return internal_coords


def find_distances(water_coor, atoms_coords):
    """
    find distances between one water and every other atoms
    ----------------------------------------------------------------------------
    water_coor: ndarray 1 x 3
    Chosen water's coordinate

    atom_coords: ndarray N x 3
    Array of protein atom coordinates

    ----------------------------------------------------------------------------
    Returns:
    dist: N x 1
    distances between given water and N prtein atoms
    """
    delta = atoms_coords - water_coor
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))

    return dist


def find_n_nearest_atoms(water, atoms, n):
    """
    find n nearest atoms near any water
    ----------------------------------------------------------------------------
    water: ndrray 1 x 7
    |A|A|R|R|X|Y|Z|
    Array of water information

    atoms: ndarray N x 7
    |A|A|R|R|X|Y|Z|
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    n_nearest_atoms_relative_xyz: n x 8
    |A|A|R|R|r|r|r|d|
    reformatted information from n nearest water molecules
    """

    dist = find_distances(water[-3:], atoms[:, -3:])
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    atoms_sorted = atoms_with_dist[atoms_with_dist[:, -1].argsort()]
    n_nearest_atoms = atoms_sorted[1:n + 1]

    return n_nearest_atoms


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndrray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    W = waters.shape[0]
    training_X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()

    return training_X


def generate_training_yes_y(W):
    """
    generate y training data for yes cases
    ----------------------------------------------------------------------------
    W: int
    Number of water molecules or yes cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(W)
    zeros = np.zeros(W)
    training_y = np.append(ones[:, np.newaxis], zeros[:, np.newaxis], axis=1)

    return training_y


def feature_encoder_atom(feature_number):
    """
    takes a number and outputs [cos(number), sin(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents atom_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [cos(number), sin(number)]

    """
    return np.array([np.cos(feature_number), np.sin(feature_number)])


def feature_encoder_residue(feature_number):
    """
    takes a number and outputs [sin(number), cos(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents residue_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [sin(number), cos(number)]

    """
    return np.array([np.sin(feature_number), np.cos(feature_number)])


def read_pdb(input_pdb):
    """
    reads a pdb file and returns numpy array of water data and protein data
    ----------------------------------------------------------------------------
    input_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    water_data, protein_data: ndarray: N x 7
    """
    # read in the pdb file
    pdb_file = open(input_pdb)
    atom_info = [line for line in pdb_file.readlines()
                 if line.startswith('ATOM  ') or line.startswith('HETATM')]
    water_data = []
    protein_data = []
    num_of_atom_types = len(atom_types.keys())
    num_of_residue_types = len(residue_types.keys())
    for line in atom_info:
        one_data = np.array([])
        xyz = [float(x) for x in line[30:53].split()]
        # read in the atom name
        atom_type = str(line[76:78]).strip()
        res_type = str(line[17:20]).strip()
        try:
            atom_encode = feature_encoder_atom(atom_types[atom_type])
        except KeyError:
            num_of_atom_types += 1
            atom_types[atom_type] = num_of_atom_types
            atom_encode = feature_encoder_atom(atom_types[atom_type])
            print("atom_types:", atom_types)
        try:
            residue_encode = feature_encoder_residue(residue_types[res_type])
        except KeyError:
            num_of_residue_types += 1
            residue_types[res_type] = num_of_residue_types
            residue_encode = feature_encoder_residue(residue_types[res_type])
            print("residue_types:", residue_types)

        one_data = np.append(one_data, atom_encode)
        one_data = np.append(one_data, residue_encode)
        one_data = np.append(one_data, xyz)
        if res_type == 'HOH':
            water_data.append(one_data)
        else:
            protein_data.append(one_data)

    return np.array(water_data), np.array(protein_data)


if __name__ == '__main__':
    input_pdb = sys.argv[1]
    water_data, protein_data = read_pdb(input_pdb)
    total_data = np.append(water_data, protein_data, axis=0)
    training_X = generate_training_yes_X(water_data, total_data, n=10)
    training_y = generate_training_yes_y(water_data.shape[0])
    np.save('test_data/CI_X.npy', training_X)
    np.save('test_data/CI_y.npy', training_y)

    pass
