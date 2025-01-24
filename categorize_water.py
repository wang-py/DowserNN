import numpy as np
import argparse
from pdb_input_processing import read_pdb
parser = argparse.ArgumentParser(
        prog='categorize_water.py',
        description='script that separates water molecules into \
                surface and internal water and output them as pdbs',
        )
parser.add_argument('-f', '--filename')
parser.add_argument('-s', '--surface_file')
parser.add_argument('-o', '--output_filename')
parser.add_argument('-r', '--include_radius')


def read_surface_points(surface_file):
    # read in the vertex file
    surface = open(surface_file)
    # skip comments
    surface_points = [line.split()[0:6] for line in surface.readlines()[3:]]
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


def is_on_surface(water_coords, surface_points, radius=3):
    surface_coor = surface_points[:, 0:3]
    surface_normal = surface_points[:, 3:]
    delta = water_coords - surface_coor
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))
    dist_check = dist <= radius
    if not dist_check.any():
        return False
    water_in_range = delta[dist_check]
    vec_in_range = surface_normal[dist_check]
    dist_along_normal = np.dot(water_in_range, vec_in_range.T)
    normal_check = dist_along_normal > 0
    if normal_check.any():
        return True

    return False


def get_surface_and_internal_water(water_data, surface_points, radius=3):
    surface_water = []
    internal_water = []
    for one_water in water_data:
        one_xyz = np.array(one_water[30:54].split()).astype(float)
        if is_on_surface(one_xyz, surface_points, radius):
            surface_water.append(one_water)
        else:
            internal_water.append(one_water)

    return surface_water, internal_water


def write_water_to_pdb(water_data, output_filename):
    with open(output_filename, 'w') as pdb:
        pdb.writelines(water_data)

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.filename)
    print(args.surface_file)
    print(args.output_filename)
    if not args.include_radius:
        args.include_radius = 3
    print(args.include_radius)
    atom_info = read_pdb(args.filename)
    surface_points = read_surface_points(args.surface_file)
    water_data = get_water_data(atom_info)
    surface_water_data, internal_water_data = get_surface_and_internal_water(
            water_data,
            surface_points,
            args.include_radius
            )
    write_water_to_pdb(surface_water_data,
                       args.output_filename + "_surface.pdb")
    write_water_to_pdb(internal_water_data,
                       args.output_filename + "_internal.pdb")
    pass
