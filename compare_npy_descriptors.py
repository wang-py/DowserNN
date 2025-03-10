import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
        prog='compare_npy_decriptors.py',
        description='script that compare 2 descriptor files',
        )
parser.add_argument('-f1', '--npy_descriptor_file1', type=str)
parser.add_argument('-f2', '--npy_descriptor_file2', type=str)
#parser.add_argument('-t', '--test_file', type=str)
#parser.add_argument('-w', '--water_pdb', type=str)
#parser.add_argument('-m', '--model', type=str)
#parser.add_argument('-k', '--sort_key', type=str)





if __name__ == "__main__":
    # Generate training and validation data
    args = parser.parse_args()
    X_file1 = args.npy_descriptor_file1
    X_file2 = args.npy_descriptor_file2

    X1 = np.load(X_file1)
    X2 = np.load(X_file2)
    np.set_printoptions(precision=4, suppress=True)

    N = len(X1)
    print(f'Number of sites in the 1st file is {N}.')
    if len(X2) != N:
        print(f'FALSE: two files have different number of sites:  {N} sites in X_file1 and {len(X2)} sites in X_file2.')
        exit()

    diff_X2_X1 = X2 - X1

    if np.all(diff_X2_X1 == 0.0):
        print(f'TRUE: Descriptors in both files are identical')
    else:
        print(f'FALSE: both files have same number of sites {N} but descriptors are different.')
        print(diff_X2_X1[0,:])
        print(diff_X2_X1[N-1,:])
        row_indices, col_indices = np.nonzero(diff_X2_X1)
        print("Row indices:", row_indices[:10])
        print("Column indices:", col_indices[:10])




