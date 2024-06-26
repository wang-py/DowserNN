import sys
import os
import numpy as np


def get_data_from_csv(csv_files):
    data = np.zeros([len(csv_files), 80])
    for i in range(len(csv_files)):
        csv_data = np.genfromtxt(csv_files[i], delimiter=',')
        data[i] = csv_data[1:3, :].flatten()

    return data


if __name__ == '__main__':
    input_folder = sys.argv[1]
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for name in files:
            if name.endswith(".csv"):
                file = os.path.join(root, name)
                csv_files.append(file)
    print(csv_files)

    data = get_data_from_csv(csv_files)
    pass
