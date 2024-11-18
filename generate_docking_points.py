import numpy as np
import sys
import os
import timeit
from pdb_input_processing import read_pdb, read_cavities
from pdb_input_processing import feature_encoder_residue
from pdb_input_processing import find_n_nearest_atoms, get_internal_coords

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21}  # , 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'SD': 4, 'H': 5, 'CA': 6, 'CB': 7,
              'CG': 8, 'CD1': 9, 'CD2': 10, 'CE1': 11, 'CE2': 12, 'CZ': 13}


def generate_docking_points(atoms, cavities, n):
    """
    Generate docking points for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in xyz

    n: int
    Number of closest atoms

    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 10
    descriptors for docking points
    """
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    docking_points = []
    for i in range(0, C):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            internal_coords = get_internal_coords(
                    n_nearest_atoms[:, -4:-1] - cavities[i, -3:])
            one_docking_point = np.append(n_nearest_atoms[:, 0:4],
                                          internal_coords, axis=1)
            # include original xyz coords
            one_docking_point = np.append(one_docking_point, cavities[i, -3:])
            docking_points.append(one_docking_point.flatten())

    return np.array(docking_points)


if __name__ == "__main__":
    try:
        input_pdb = sys.argv[1]
        input_cavities = sys.argv[2]
    except IndexError:
        print("Usage: python generate_docking_points.py input_pdb input_cavities")
        exit()
    pdb_name = os.path.basename(input_pdb).split('.')[0]
    water_data, protein_data = read_pdb(input_pdb)
    cavities_data = read_cavities(input_cavities)
    print("Generating docking points...")
    starting_time = timeit.default_timer()
    docking_points = generate_docking_points(protein_data, cavities_data, n=10)
    ending_time = timeit.default_timer()
    print("number of docking points: %d" % docking_points.shape[0])
    total_time = ending_time - starting_time
    print(f"Docking initialization took {total_time:.2f} seconds")
    np.save(f'test_data/{pdb_name}_docking_points.npy', docking_points)
