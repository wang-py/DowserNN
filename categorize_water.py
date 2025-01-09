import numpy as np
import argparse
from pdb_input_processing import read_pdb
parser = argparse.ArgumentParser(prog='categorize_water.py',
                                 description='script that separates water molecules into surface and internal water and output them as pdbs',
                                 )
parser.add_argument('filename')
parser.add_argument('surface')


def read_surface_points(surface_file):
    # read in the vertex file
    surface = open(surface_file)
    # skip comments
    surface_points = [line.split()[0:3] for line in surface.readlines()[3:]]
    surface_points = np.array(surface_points)

    return surface_points


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.filename)
    print(args.surface)
    atom_info = read_pdb(args.filename)
    surface_points = read_surface_points(args.surface)
    pass
