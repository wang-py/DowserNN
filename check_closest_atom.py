import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from glob import glob
import time
import os.path


def plot_distance_hist(bins, this_xlabel, distribution, save=False):
    # font size
    font_size = 11

    # plotting distribution distribution
    fig, ax = plt.subplots()
    ax.set(title='Distribution of shortest distance from water')
    if this_xlabel:
        ax.set(xlabel=this_xlabel + ' distribution [A]')
        ax.xaxis.label.set_size(font_size)
    # ax.set(ylabel = 'Frequency')
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # crystal structure line
    # get mean of distribution
    distribution_mean = np.mean(distribution)
    distribution_mean_str = f"{distribution_mean:.2f}"
    ax.axvline(distribution_mean, color='k', linestyle='--',
               label='mean = ' + distribution_mean_str + ' A')
    # histogram with colors
    N, bin_min, patches = ax.hist(distribution, bins, density=True)
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax.legend(loc='best')
    if save:
        plt.savefig(this_xlabel+".png", dpi=200)
    # plt.show()


def parse_pdb_file(file_path):
    waters = []
    atoms = []
    unique_waters = {}
    with open(file_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("HETATM") and "HOH" in line[17:20]:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                residue_seq = line[20:26]
                if residue_seq not in unique_waters:
                    unique_waters[residue_seq] = {
                        "x": x,
                        "y": y,
                        "z": z,
                        # "residue_seq": residue_seq,
                    }
            elif (
                line.startswith("ATOM")
                    or (line.startswith("HETATM"))):  # and "HOH" not in line[17:20])
            # ) and line[76:78].strip() != "H":
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append({"x": x, "y": y, "z": z})
    waters = list(unique_waters.values())
    return np.array(waters), np.array(atoms)


def find_distances(water_coor, atoms_coords):
    delta = water_coor - atoms_coords
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))

    return dist


def find_closest_distances(waters, atoms):
    water_coors = np.array([np.fromiter(water.values(), dtype=float)[0:3] for
                            water in waters])
    water_coors = water_coors.astype(float)
    atoms_coords = np.array([np.fromiter(atom.values(), dtype=float) for
                             atom in atoms])
    min_distances = []
    for water_coor in water_coors:
        dist = find_distances(water_coor, atoms_coords)
        min_distances.append(dist.min())

    return min_distances


def find_nth_closest_distances(waters, atoms, n):
    water_coors = np.array([np.fromiter(water.values(), dtype=float)[0:3] for
                            water in waters])
    water_coors = water_coors.astype(float)
    atoms_coords = np.array([np.fromiter(atom.values(), dtype=float) for
                             atom in atoms])
    nth_min_distances = []
    for water_coor in water_coors:
        dist = find_distances(water_coor, atoms_coords)
        sorted_dist = np.sort(dist)
        nth_min_distances.append(sorted_dist[n - 1])

    return nth_min_distances
# def find_closest_distances(waters, atoms):
#     distances = []
#     for water in waters:
#         min_distance = float("inf")
#         for atom in atoms:
#             distance = calculate_distance(water, atom)
#             if distance < min_distance:
#                 min_distance = distance
#         distances.append(min_distance)
#     return distances


def plot_distances(waters, distances, title, savefig=False):
    # residue_seq_numbers = [water["residue_seq"] for water in waters]
    arbitrary_order = list(range(1, len(waters) + 1))
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    plt.figure(figsize=(14, 7))
    plt.scatter(arbitrary_order, distances, color="b", label="Distances")
    plt.axhline(
        avg_distance, color="g", linestyle="--",
        label=f"Average: {avg_distance:.2f} Å"
    )
    plt.axhline(
        min_distance, color="r", linestyle="--",
        label=f"Min: {min_distance:.2f} Å"
    )
    plt.axhline(
        max_distance, color="y", linestyle="--",
        label=f"Max: {max_distance:.2f} Å"
    )

    # for i, txt in enumerate(residue_seq_numbers):
    #     plt.annotate(txt, (arbitrary_order[i], distances[i]), fontsize=8,
    #                  ha="center")

    plt.xlabel("Water Molecule")
    plt.ylabel("Closest Distance to other Atoms (Å)")
    plt.title("Closest Distance from Water Molecules other Atoms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(title + "_distance.png", dpi=200)
    # plt.show()


def get_all_nth_closest_distances(input_pdbs, n):
    all_closest_distances = np.array([])
    for one_pdb in input_pdbs:
        pdb_filename = one_pdb.split('/')[1].split('.')[0]
        print(f"Parsing PDB file {pdb_filename}...")
        waters, atoms = parse_pdb_file(one_pdb)
        print(
            f"Found {len(waters)} unique water molecules and {len(atoms)}" +
            " non-water atoms (excluding hydrogens)."
        )
        distance_calc_start = time.time()
        print("Calculating closest distances...")
        closest_distances = find_nth_closest_distances(waters, atoms, n)
        distance_calc_end = time.time()
        distance_calc_time = distance_calc_end - distance_calc_start
        print(f"distance calculation took {distance_calc_time:.2f} seconds")
        # plot_distances(waters, closest_distances, pdb_filename, savefig=False)
        # plot_distance_hist(20, pdb_filename + "_hist", closest_distances,
        #                    save=False)
        all_closest_distances = np.append(all_closest_distances,
                                          closest_distances)

    return all_closest_distances


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_water_distances.py <pdb_file_path>")
        sys.exit(1)

    pdb_file_path = sys.argv[1]
    pdb_files = glob(pdb_file_path + "/*.pdb")

    if not os.path.isfile('all_closest_distances.txt'):
        all_closest_distances = get_all_nth_closest_distances(pdb_files, n=2)
        total_num_of_water = all_closest_distances.shape[0]
        print(f"Found {total_num_of_water} water molecules")
        print("Plotting distances...")
        np.savetxt("all_closest_distances.txt", all_closest_distances,
                   fmt='%.3f')
    else:
        all_closest_distances = np.loadtxt('all_closest_distances.txt')
    # plot_distances(waters, closest_distances)
    # print("Plotting distributions...")
    plot_distance_hist(60, "2nd shortest distance histogram",
                       all_closest_distances, save=True)
    plot_distances(all_closest_distances, all_closest_distances,
                   "2nd shortest distances", savefig=True)
