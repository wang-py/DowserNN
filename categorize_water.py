import numpy as np
import argparse
from pdb_input_processing import read_pdb, find_distances
parser = argparse.ArgumentParser(prog='categorize_water.py',
                                 description='script that separates water molecules into surface and internal water and output them as pdbs',
                                 )
parser.add_argument('-f', '--filename')
parser.add_argument('-s', '--surface_file')
parser.add_argument('-o', '--surface_water')
parser.add_argument('-r', '--include_radius')


def read_surface_points(surface_file):
    # read in the vertex file
    surface = open(surface_file)
    # skip comments
    surface_points = [line.split()[0:3] for line in surface.readlines()[3:]]
    surface_points = np.array(surface_points).astype(float)

    return surface_points


def get_water_data(atom_info):
    water_data = []
    for line in atom_info:
        atom_type = str(line[13:16]).strip()
        res_type = str(line[17:20]).strip()
        if res_type == 'HOH' or atom_type == 'OW':
            water_data.append(line)

    return water_data


def get_surface_water(water_data, surface_points, radius=3):
    surface_water = []
    for one_water in water_data:
        one_xyz = np.array(one_water[30:54].split()).astype(float)
        dist = find_distances(one_xyz, surface_points)
        check = dist < radius
        if check.any():
            surface_water.append(one_water)

    return surface_water


def write_surface_water_to_pdb(surface_water_data, output_filename):
    with open(output_filename, 'w') as pdb:
        pdb.writelines(surface_water_data)

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.filename)
    print(args.surface_file)
    print(args.surface_water)
    if not args.include_radius:
        args.include_radius = 3
    print(args.include_radius)
    atom_info = read_pdb(args.filename)
    surface_points = read_surface_points(args.surface_file)
    water_data = get_water_data(atom_info)
    surface_water_data = get_surface_water(water_data, surface_points, args.include_radius)
    write_surface_water_to_pdb(surface_water_data, args.surface_water)
    pass
