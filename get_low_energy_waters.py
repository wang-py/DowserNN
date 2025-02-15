from pdb_input_processing import read_pdb
import numpy as np
import argparse
parser = argparse.ArgumentParser(
        prog='get_low_energy_waters.py',
        description='script that finds low energy waters and outputs to file',
        )
parser.add_argument('-d', '--dowser_pdb', type=str)
parser.add_argument('-g', '--gromacs_pdb', type=str)
parser.add_argument('-o', '--output_filename', type=str)


def categorize_by_cutoff(dowser_data, EM_data, cutoff=-4):
    dowser_data = np.array([float(x[60:67]) for x in dowser_data])
    EM_data = np.array(EM_data)
    low_indices = np.where(dowser_data <= cutoff)[0]
    EM_data_low_E = EM_data[low_indices]
    high_indices = np.where(dowser_data > cutoff)[0]
    EM_data_high_E = EM_data[high_indices]

    return EM_data_low_E, EM_data_high_E


def write_water_to_file(output_filename: str, low_E, high_E):
    np.savetxt(output_filename + '_low_E.pdb', low_E, fmt='%s', newline='')
    np.savetxt(output_filename + '_high_E.pdb', high_E, fmt='%s', newline='')


if __name__ == '__main__':
    args = parser.parse_args()
    dowser_data = read_pdb(args.dowser_pdb)
    EM_data = read_pdb(args.gromacs_pdb)
    EM_low_E, EM_high_E = categorize_by_cutoff(dowser_data, EM_data, cutoff=-4)
    write_water_to_file(args.output_filename, EM_low_E, EM_high_E)
    pass
