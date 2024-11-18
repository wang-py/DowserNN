import numpy as np
import matplotlib.pyplot as plt
from keras import saving
import sys


def dock_water(cavities_xyz, docking_result):
    with open("docking_result.pdb", 'w') as d:
        for i in range(len(cavities_xyz)):
            x = cavities_xyz[i][0]
            y = cavities_xyz[i][1]
            z = cavities_xyz[i][2]
            d.write(f"ATOM {i:d} {x} {y} {z} {docking_result[i][0]}\n")


if __name__ == "__main__":
    cavities = np.load(sys.argv[1])
    model = saving.load_model("test_data/DowserNN.keras")
    cavities_xyz = cavities[:, -3:]
    docking_result = model.predict(cavities[:, :-3])
    plt.plot(docking_result[:, 0])
    plt.show()
    print(docking_result)
    dock_water(cavities_xyz, docking_result)
