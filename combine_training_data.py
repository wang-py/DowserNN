import numpy as np
import sys

if __name__ == '__main__':
    input_data = sys.argv
    output_data = np.load(sys.argv[1])
    for data in input_data[2:]:
        input_npy = np.load(data)
        output_data = np.concatenate((output_data, input_npy), axis=0)
    np.save("test_data/combined_CI_y_yes.npy", output_data)
