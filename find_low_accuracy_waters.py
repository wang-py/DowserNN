import numpy as np
import os
import sys


def get_low_accuracy_waters(input_pdb, water_index_file):
    pdb = open(input_pdb)
    water_index = np.loadtxt(water_index_file, dtype=int)
    water_info = [line for line in pdb.readlines() if
                  line.startswith('HETATM') and 'HOH' in line]
    low_accuracy_waters = []
    for i in water_index:
        low_accuracy_waters.append(water_info[i])

    return low_accuracy_waters


def write_low_accuracy_waters(low_accuracy_waters, output_pdb):
    with open(output_pdb, 'w') as out_pdb:
        for line in low_accuracy_waters:
            out_pdb.write(line)

    pass


if __name__ == '__main__':
    input_pdb = sys.argv[1]
    pdb_filename = os.path.basename(input_pdb)
    water_index_file = sys.argv[2]

    try:
        output_pdb = sys.argv[3]
    except IndexError:
        output_pdb = pdb_filename.split('.')[0] + "_low_accuracy_waters.pdb"

    low_accuracy_waters = get_low_accuracy_waters(input_pdb, water_index_file)

    write_low_accuracy_waters(low_accuracy_waters, output_pdb)
    pass
