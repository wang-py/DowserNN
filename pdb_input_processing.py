import sys
import os
import numpy as np

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21, 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'S': 4, 'H': 5, 'Others': 6}


def feature_encoder(feature_number):
    """
    takes a number and outputs [cos(number), sin(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents atom_types or residue_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [cos(number), sin(number)]

    """
    return np.array([np.cos(feature_number), np.sin(feature_number)])


def read_pdb(input_pdb):
    # read in the pdb file
    tunnel_points = []
    pdb_file = open(input_pdb)
    atom_info = [line for line in pdb_file.readlines()
                 if line.startswith('ATOM  ') or line.startswith('HETATM')]
    for line in atom_info:
        xyz = [float(x) for x in line[30:53].split()]
        # read in the atom name
        atom_type = str(line[12:15]).strip()
        residue_type = str(line[17:19]).strip()
        try:
            atom_encode = feature_encoder(atom_types[atom_type])
            residue_encode = feature_encoder(residue_types[residue_type])
        except KeyError as err:
            atom_types.setdefault(atom_type, 6)
            residue_types.setdefault(residue_type, 22)

        tunnel_points.append(xyz)

    return np.array(tunnel_points)


if __name__ == '__main__':
    input_pdb = sys.argv[1]
    read_pdb(input_pdb)

    pass
