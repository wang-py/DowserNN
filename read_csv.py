import sys
import os
import numpy as np


def get_data_from_csv(csv_files):
    X = np.zeros([len(csv_files), 80])
    y = np.zeros(len(csv_files))
    for i in range(len(csv_files)):
        csv_data = np.genfromtxt(csv_files[i], delimiter=',')
        X[i] = csv_data[3:5, :].flatten()
        if csv_files[i].endswith('_1.csv'):
            y[i] = 1

    return X, y


if __name__ == '__main__':
    input_folder = sys.argv[1]
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for name in files:
            if name.endswith(".csv"):
                file = os.path.join(root, name)
                csv_files.append(file)
    print(csv_files)

    X, y = get_data_from_csv(csv_files)
    np.save(input_folder + "/X.npy", X)
    np.save(input_folder + "/y.npy", y)
    pass
