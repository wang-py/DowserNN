import numpy as np
import sys
import matplotlib.pyplot as plt


def get_xyz(pdb_file):
    with open(pdb_file, 'r') as pdb:
        data = pdb.readlines()
        xyz = np.array([[float(line[30:38]),
                         float(line[38:46]),
                         float(line[46:54])] for line in data])
    return xyz


def get_rmsd(xyz1, xyz2):
    rmsd = np.mean(np.sqrt(np.sum(np.square(xyz1 - xyz2), axis=1)))
    print(f"total rmsd is: {rmsd: .2f} A")
    return rmsd


def get_distance(xyz1, xyz2):
    dist = np.sqrt(np.sum(np.square(xyz1 - xyz2), axis=1))
    return dist


def plot_rmsd(rmsd):

    pass


def plot_dist(dist):
    water_i = np.arange(len(dist)) + 1
    avg_dist = np.mean(dist)
    plt.title("water drift after EM")
    plt.xlabel("water index")
    plt.ylabel("distance [A]")
    plt.plot(water_i, dist)
    plt.axhline(avg_dist, color='k', linestyle='--',
                label=f"average distance: {avg_dist: .2f}")
    plt.legend()
    plt.show()
    pass


if __name__ == "__main__":
    input_pdb = sys.argv[1]
    compare_pdb = sys.argv[2]

    xyz1 = get_xyz(input_pdb)
    xyz2 = get_xyz(compare_pdb)

    rmsd = get_rmsd(xyz1, xyz2)

    dist = get_distance(xyz1, xyz2)

    plot_dist(dist)


    pass
