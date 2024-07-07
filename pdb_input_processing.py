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
    dist: ndarray N x 1
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


def find_atoms_within_box(x, step_x, y, step_y, z, step_z, atoms):
    scan_x = np.logical_and(atoms[:, -3] < (x + step_x), atoms[:, -3] >= x)
    scan_y = np.logical_and(atoms[:, -2] < (y + step_y), atoms[:, -2] >= y)
    scan_z = np.logical_and(atoms[:, -1] < (z + step_z), atoms[:, -1] >= z)
    atoms_within_box = np.logical_and(scan_x, scan_y)
    atoms_within_box = np.logical_and(atoms_within_box, scan_z)

    return atoms_within_box


def get_input_partitions(atoms, partitions=2):
    """
    calculate partitions for protein atoms; the size is defined by box_size
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of protein atoms' information

    box_size: float angstoms
    Size of a single partition
    ----------------------------------------------------------------------------
    Returns:
    atoms_partitions: ndarray num_of_boxes x atoms_in_box x 7
    """
    box_min = np.min(atoms[:, -3:], axis=0)
    box_max = np.max(atoms[:, -3:], axis=0)
    # box_length = box_max - box_min
    # num_of_boxes = (box_length / box_size).astype(int)
    partitions_x, step_x = np.linspace(box_min[0], box_max[0], partitions,
                                       retstep=True)
    partitions_y, step_y = np.linspace(box_min[1], box_max[1], partitions,
                                       retstep=True)
    partitions_z, step_z = np.linspace(box_min[2], box_max[2], partitions,
                                       retstep=True)
    atoms_partitions = []
    for one_x in partitions_x:
        for one_y in partitions_y:
            for one_z in partitions_z:
                atoms_within_box = find_atoms_within_box(one_x, step_x,
                                                         one_y, step_y,
                                                         one_z, step_z,
                                                         atoms)
                if atoms_within_box.any():
                    atoms_partitions.append(atoms[atoms_within_box])

    return atoms_partitions


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
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


def check_num_of_protein_atoms(atoms_partitions, atoms):
    num_of_atoms_in_partitions = 0
    num_of_atoms = atoms.shape[0]
    for i in range(len(atoms_partitions)):
        one_P = atoms_partitions[i].shape[0]
        print(f"num of atoms in partition {i + 1}: {one_P}")
        num_of_atoms_in_partitions += one_P

    if num_of_atoms_in_partitions == num_of_atoms:
        return True

    print(f"there are {num_of_atoms_in_partitions} atoms in partitions")
    return False


def generate_training_no_X(atoms, n):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    print("Partitioning protein atoms...")
    atoms_partitions = get_input_partitions(atoms, partitions=2)
    print("Checking atom count in partitions...")
    is_same_count = check_num_of_protein_atoms(atoms_partitions, atoms)
    if is_same_count:
        print("Partitioning successful")
    else:
        print("Atom count error")
        exit()
    num_of_partitions = len(atoms_partitions)
    P = atoms.shape[0]
    training_X = []  # np.zeros([None, 7 * n])
    for i in range(num_of_partitions):
        partition = atoms_partitions[i]
        num_atoms_in_box = partition.shape[0]
        for j in range(num_atoms_in_box):
            n_nearest_atoms = find_n_nearest_atoms(partition[j], partition, n)
            internal_coords = get_internal_coords(
                    n_nearest_atoms[:, -4:-1] - atoms[i, -3:])
            one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                       internal_coords, axis=1)
            training_X.append(one_training_X.flatten())

    if len(training_X) == P:
        return training_X
    else:
        print("Number of training X data and input mismatch")
        exit()


def generate_training_no_y(P):
    """
    generate y training data for no cases
    ----------------------------------------------------------------------------
    P: int
    Number of protein atoms or no cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(P)
    zeros = np.zeros(P)
    training_y = np.append(zeros[:, np.newaxis], ones[:, np.newaxis], axis=1)

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
            # print("atom_types:", atom_types)
        try:
            residue_encode = feature_encoder_residue(residue_types[res_type])
        except KeyError:
            num_of_residue_types += 1
            residue_types[res_type] = num_of_residue_types
            residue_encode = feature_encoder_residue(residue_types[res_type])
            # print("residue_types:", residue_types)

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
    training_yes_X = generate_training_yes_X(water_data, total_data, n=10)
    training_yes_y = generate_training_yes_y(water_data.shape[0])
    training_no_X = generate_training_no_X(protein_data, n=10)
    training_no_y = generate_training_no_y(protein_data.shape[0])
    training_X = np.append(training_yes_X, training_no_X, axis=0)
    training_y = np.append(training_yes_y, training_no_y, axis=0)
    # TODO: need to randomize X and y arrays
    # np.random.shuffle(training_X)
    # np.random.shuffle(training_y)
    np.save('test_data/CI_X_test.npy', training_X)
    np.save('test_data/CI_y_test.npy', training_y)

    pass
