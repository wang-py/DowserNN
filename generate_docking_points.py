import numpy as np
import sys
import os
import timeit
from pdb_input_processing import read_pdb, read_cavities
from pdb_input_processing import generate_training_no_X

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21}  # , 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'SD': 4, 'H': 5, 'CA': 6, 'CB': 7,
              'CG': 8, 'CD1': 9, 'CD2': 10, 'CE1': 11, 'CE2': 12, 'CZ': 13}

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
    total_data = np.append(water_data, protein_data, axis=0)
    print("Generating docking points...")
    starting_time = timeit.default_timer()
    docking_points = generate_training_no_X(total_data, cavities_data, n=10,
                                            interval=1)
    ending_time = timeit.default_timer()
    print("number of docking points: %d" % docking_points.shape[0])
    total_time = ending_time - starting_time
    print(f"Docking initialization took {total_time:.2f} seconds")
    np.save(f'test_data/{pdb_name}_docking_points.npy', docking_points)
