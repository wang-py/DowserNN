import numpy as np
import matplotlib.pyplot as plt
from keras import saving
import sys


def format_water_oxygen_xyz(i, xyz, confidence):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    entry = f"ATOM  {i+1:d>4}  OW  HOH D 1   {x:8.3f>7}{y:8.3f>7}{z:8.3f>7}" +\
        f" {confidence:6.2f>5}" + f"{'':<10}" + "O"
    return entry


def dock_water(cavities_xyz, docking_result):
    with open("docking_result.pdb", 'w') as d:
        for i in range(len(cavities_xyz)):
            d.write(format_water_oxygen_xyz(i, cavities_xyz,
                                            docking_result[i]))


if __name__ == "__main__":
    cavities = np.load(sys.argv[1])
    model = saving.load_model("test_data/DowserNN.keras")
    cavities_xyz = cavities[:, -3:]
    docking_result = model.predict(cavities[:, :-3])
    plt.plot(docking_result[:, 0])
    plt.show()
    print(docking_result)
    dock_water(cavities_xyz, docking_result)
