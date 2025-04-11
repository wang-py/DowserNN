import sys
import os
import timeit
import numpy as np

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21}  # , 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'SD': 4, 'H': 5, 'CA': 6, 'CB': 7,
              'CG': 8, 'CD1': 9, 'CD2': 10, 'CE1': 11, 'CE2': 12, 'CZ': 13}


def get_internal_coords(relative_coors, scaling_factor: float = 100):
    """
    calculate internal coordinates based on relative vectors
    ----------------------------------------------------------------------------
    relative_coors: ndarray N x 3
    Array of relative coordinates from water to any other atom
    ----------------------------------------------------------------------------
    Returns:
    internal_coords: ndarray N x 3
    internal coordinates based on relative positions
    """
    N = relative_coors.shape[0]
    internal_coords = np.zeros([N, 3])
    r1 = relative_coors[0, :]
    r2 = relative_coors[1, :]
    internal_coords[0] = np.array([r1.dot(r1.T), 0, 0])
    internal_coords[1] = np.array([r2.dot(r2.T), r2.dot(r1.T), 0])

    for i in range(2, N):
        r_a = relative_coors[i]
        r_b = relative_coors[i - 1]
        r_c = relative_coors[i - 2]
        internal_coords[i] = np.array([r_a.dot(r_a.T),
                                       r_a.dot(r_b.T),
                                       r_a.dot(r_c.T)])

    return internal_coords / scaling_factor


def find_distances(water_coor, atoms_coords):
    """
    find distances between one water and every other atoms
    ----------------------------------------------------------------------------
    water_coor: ndarray 1 x 3
    Chosen water's coordinate

    atom_coords: ndarray N x 3
    Array of protein atom coordinates

    ----------------------------------------------------------------------------
    Returns:
    dist: ndarray N x 1
    distances between given water and N prtein atoms
    """
    delta = atoms_coords - water_coor
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))

    return dist


def find_n_nearest_atoms(water, atoms, n):
    """
    find n nearest atoms near any water
    ----------------------------------------------------------------------------
    water: ndrray 1 x 7
    |A|A|R|R|X|Y|Z|
    Array of water information

    atoms: ndarray N x 7
    |A|A|R|R|X|Y|Z|
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    n_nearest_atoms_relative_xyz: n x 8
    |A|A|R|R|r|r|r|d|
    reformatted information from n nearest water molecules
    """

    dist = find_distances(water[-3:], atoms[:, -3:])
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    search_range = 15
    atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] <
                                                  search_range)]
    if atoms_within_range.shape[0] >= n:
        atoms_sorted = atoms_within_range[atoms_within_range[:, -1].argsort()]
    # if there are not enough atoms within range
    else:
        atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] <
                                                      search_range * 2)]
    n_nearest_atoms = atoms_sorted[1:n + 1]
    if atoms_sorted[0,-1] > 0.0:   # !!! If nearest atoms are searched not for atoms
            n_nearest_atoms = atoms_sorted[0:n]
    return n_nearest_atoms


def find_atoms_within_box(x, step_x, y, step_y, z, step_z, atoms):
    scan_x = np.logical_and(atoms[:, -3] < (x + step_x), atoms[:, -3] >= x)
    scan_y = np.logical_and(atoms[:, -2] < (y + step_y), atoms[:, -2] >= y)
    scan_z = np.logical_and(atoms[:, -1] < (z + step_z), atoms[:, -1] >= z)
    atoms_within_box = np.logical_and(scan_x, scan_y)
    atoms_within_box = np.logical_and(atoms_within_box, scan_z)

    return atoms_within_box

#============================= My Modified =============================
def dump_pdb(XYZ, title,pdb_path, idx_at = []):
    """
    write XYZ coordinates to pdb file
    ----------------------------------------------------------------------------
    title - description of the XYZ coordinates to be saved in pdb Title
    XYZ ndarray N x 3 - casrtessian coordinates in Angstroms
    pdb_fname: str - path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    NO
    """
    try:
        f = open(pdb_path, 'w')
    except OSError:
        print(f"Error: cannot open file for writing {pdb_path}\nExit")
        exit()

    len_idx = len(idx_at)
    if len_idx == 0:  # Use default order indecies if idx_at is not specified  
        idx_at = range(1, len(XYZ) + 1)
        len_idx = len(idx_at)
    check_int = all(isinstance(elem, int) for elem in idx_at[1:])
    if (not check_int or len_idx != len(XYZ)):
        print(f"Error in dump_pdb: the atom index array is not correct check_int = {check_int}, len_idx = {len_idx}\nExit")
        exit()

    lines = ['TITLE     ' + title + '\n']
    # PDB Format examples: https://github.com/biopython/biopython/blob/master/Bio/PDB/PDBIO.py
    #                      https://cupnet.net/pdb-format/
    # HETATM    1  XP  CAV X   1     263.000 193.000 345.000  1.00  5.00           X  
    _PDB_ATOM_FORMAT_STRING = ('%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n')
    for i in range(len(XYZ)):
        #      (record_type,atom_number,name,altloc,resname,chain_id,resseq,icode,       x,       y,       z,occupancy,bfactor,segid,element,charge)
        args = ('HETATM',(idx_at[i]+1) % 100000,'XP',   ' ',  'CAV',     'X', (i+1)%10000,  ' ',XYZ[i,0],XYZ[i,1],XYZ[i,2],      1.0,    1.0,  ' ',    'X',   ' ')
        lines.append(_PDB_ATOM_FORMAT_STRING % args)

    f.writelines(lines)                    # Read lines with "\n" at the end
    f.close()


def atoms_within_cutoff(water, atoms, cutoff):
    """
    find atoms within cutoff for each water
    ----------------------------------------------------------------------------
    water: ndrray 1 x 7
    |A|A|R|R|X|Y|Z|
    Array of water information

    atoms: ndarray N x 7
    |A|A|R|R|X|Y|Z|
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    n_nearest_atoms_relative_xyz: n x 8
    |A|A|R|R|r|r|r|d|
    reformatted information from n nearest water molecules
    """

    dist = find_distances(water[-3:], atoms[:, -3:])
    # print(f'water[-3:]: {water[-3:]}')
    # print(f'atoms[:5,-3:]: {atoms[:5,-3:]}')
    # print(f'distance: {dist[0:10]}')
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] < cutoff)]
    env_atoms_within_range = atoms_within_range[np.where(atoms_within_range[:, -1] != 0.0)]
    return env_atoms_within_range

def Fc(r, cutoff):
    if r < cutoff:
        return 0.5*(1 + np.cos(np.pi * r/cutoff) )
    else:
        return 0

def generate_AEV_descriptors_yes_X_0(waters, atoms):
    """
    Generate X training data for yes cases for neural network
    paper: Isaev, Roitberg_ANI-1_2017
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_radial = 5.0
    cutoff_angular = 3.5
    # Parameters for Radial descriptors AEVs
    sr  = [1.0, 0.5, 0.25]
    Rs  = [2.0, 2.7, 3.5, 4.5]
    # Parameters for Angular descriptors AEVs
    sA  = [0.5, 0.25]
    RsA = [2.0, 2.7, 3.2]
    pTs = [1.0, 2.0]
    Ts  = [0.0, np.pi * 1.0 / 3.0, np.pi * 2.0/3.0, np.pi]


    W = waters.shape[0]
    N_descriptors = len(sr) * len(Rs)   +    len(sA) * len(RsA) * len(pTs) * len(Ts)
    training_X = np.zeros([W, N_descriptors])
    #training_X = np.zeros([W, 7 * n])
    print(f'Number of descriptors per water molecule: {N_descriptors}')
    for i in range(W):
        enviroment_atoms_r = atoms_within_cutoff(waters[i],              atoms, cutoff_radial)
        enviroment_atoms_a = atoms_within_cutoff(waters[i], enviroment_atoms_r[:,:-1], cutoff_angular)
        #print(f'enviroment_atoms_r: dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,:]}')
        #print(f'waters[:2]: \n{waters[:2]}')
        G = np.zeros(N_descriptors)
        # Radial descriptors
        for j in range(enviroment_atoms_r.shape[0]):
            Rij = enviroment_atoms_r[j,-1]
            Fc_ij = Fc(Rij, cutoff_angular)
            m = 0
            for s in sr:
               w = 1 / (2*s*s)
               for R0 in Rs:
                   #Gterm = np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
                   #print(f'water {i}, m={m}, s={s}, Rs={R0}: Rij={Rij} Fc_ij = {Fc_ij} Gterm = {Gterm}')
                   G[m] = G[m] + np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
                   m = m+1
        #print(f'Radial descriptors for water {i}: {G[0:11]}')
        #continue
        #exit()
        m0A = m
        # print(f'waters[{i}]: =============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,-1]}')
        # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\n{enviroment_atoms_a[:,-1]}')
        # Angular descriptors
        for j in range(enviroment_atoms_a.shape[0]):
            Rij = enviroment_atoms_a[j,-1]
            Fc_ij = Fc(Rij, cutoff_angular)
            rij = np.array( enviroment_atoms_a[j,-4:-1] - waters[i, -3:] )
            for k in range(enviroment_atoms_a.shape[0]):
                Rik = enviroment_atoms_a[k,-1]
                Fc_ik = Fc(Rik, cutoff_angular)
                rik = np.array( enviroment_atoms_a[k,-4:-1] - waters[i, -3:] )
                cos_theta = rij.dot(rik.T) / Rij / Rik
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range
                theta_ijk = np.arccos(cos_theta)
                Fc_ij_x_Fc_ik = Fc_ij  * Fc_ik
                Rij_Rjk = 0.5*(Rij+Rik)
                m = m0A
                #print(f'water {i}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} cos_theta={cos_theta} theta_ijk = {theta_ijk}')
                for s in sA:
                  w = 1 / (2*s*s)
                  for R0 in RsA:
                    fac_r = np.exp( -w * (Rij_Rjk - R0)*(Rij_Rjk - R0) ) * Fc_ij_x_Fc_ik
                    for p in pTs:
                      fac_p = 2 ** (1 - p)
                      fac = fac_r * fac_p
                      for T0 in Ts:
                        G[m] = G[m] + fac * (1 + np.cos(theta_ijk - T0))**p 
                        #Gterm = fac * (1 + np.cos(theta_ijk - T0))**p
                        #print(f'water {i}, m={m}, s={s}, Rs={R0}, p={p}, T0={T0}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} theta_ijk = {theta_ijk}, Gterm = {Gterm}')
                        #G[m] = G[m] + np.exp( -w * (Rij_Rjk - R0)^2 ) * Fc_ij_x_Fc_ik * np.power(1 + np.cos(theta_ijk - T0), p)
                        m = m+1
        training_X[i] = G
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        if i > 2940:
            print(f'waters[{i}]: =============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\nDistances:\n{enviroment_atoms_r[:,-1]}')
            print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\nDistances:\n{enviroment_atoms_a[:,-1]}')
            print(f' Descriptors for water {i}:\n{training_X[i]}')
    exit()
    return training_X

def generate_AEVs_X(center, atoms):
    """
    Generate X training data for yes cases for neural network
    paper: Isaev, Roitberg_ANI-1_2017
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_radial = 5.0
    cutoff_angular = 3.5
    # Parameters for Radial descriptors AEVs
    sr  = [1.0, 0.5, 0.25]
    Rs  = [2.0, 2.7, 3.5, 4.5]
    # Parameters for Angular descriptors AEVs
    sA  = [0.5, 0.25]
    RsA = [2.0, 2.7, 3.2]
    pTs = [1.0, 2.0]
    Ts  = [0.0, np.pi * 1.0 / 3.0, np.pi * 2.0/3.0, np.pi]


    N_descriptors = len(sr) * len(Rs)   +    len(sA) * len(RsA) * len(pTs) * len(Ts)
    G = np.zeros(N_descriptors)
    #print(f'Number of descriptors per water molecule: {N_descriptors}')
    enviroment_atoms_r = atoms_within_cutoff(center,                     atoms, cutoff_radial)
    enviroment_atoms_a = atoms_within_cutoff(center, enviroment_atoms_r[:,:-1], cutoff_angular)
    # print(f'center:\n{center}')
    # print(f'=============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,-1]}')
    # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\n{enviroment_atoms_a[:,-1]}')

    # Radial descriptors
    m = 0
    for j in range(enviroment_atoms_r.shape[0]):
        Rij = enviroment_atoms_r[j,-1]
        Fc_ij = Fc(Rij, cutoff_radial)
        m = 0
        for s in sr:
           w = 1 / (2*s*s)
           for R0 in Rs:
               #Gterm = np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
               #print(f'water {i}, m={m}, s={s}, Rs={R0}: Rij={Rij} Fc_ij = {Fc_ij} Gterm = {Gterm}')
               G[m] = G[m] + np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
               m = m+1
    #print(f'Radial descriptors for water {i}: {G[0:11]}')
    #continue
    #exit()
    m0A = m
#=======================================================================
#    if np.all(G[0:m0A-1] == 0.0):
#        print('-------------')
#        print(f'ERROR: found ZERO radial descriptors.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]}')
#=======================================================================
    # Angular descriptors
    for j in range(enviroment_atoms_a.shape[0]):
        Rij = enviroment_atoms_a[j,-1]
        Fc_ij = Fc(Rij, cutoff_angular)
        rij = np.array( enviroment_atoms_a[j,-4:-1] - center[-3:] )
        for k in range(enviroment_atoms_a.shape[0]):
            Rik = enviroment_atoms_a[k,-1]
            Fc_ik = Fc(Rik, cutoff_angular)
            rik = np.array( enviroment_atoms_a[k,-4:-1] - center[-3:] )
            cos_theta = rij.dot(rik.T) / Rij / Rik
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range
            theta_ijk = np.arccos(cos_theta)
            Fc_ij_x_Fc_ik = Fc_ij  * Fc_ik
            Rij_Rjk = 0.5*(Rij+Rik)
            m = m0A
            #print(f'water {i}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} cos_theta={cos_theta} theta_ijk = {theta_ijk}')
            for s in sA:
              w = 1 / (2*s*s)
              for R0 in RsA:
                fac_r = np.exp( -w * (Rij_Rjk - R0)*(Rij_Rjk - R0) ) * Fc_ij_x_Fc_ik
                for p in pTs:
                  fac_p = 2 ** (1 - p)
                  fac = fac_r * fac_p
                  for T0 in Ts:
                    G[m] = G[m] + fac * (1 + np.cos(theta_ijk - T0))**p 
                    #Gterm = fac * (1 + np.cos(theta_ijk - T0))**p
                    #print(f'water {i}, m={m}, s={s}, Rs={R0}, p={p}, T0={T0}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} theta_ijk = {theta_ijk}, Gterm = {Gterm}')
                    #G[m] = G[m] + np.exp( -w * (Rij_Rjk - R0)^2 ) * Fc_ij_x_Fc_ik * np.power(1 + np.cos(theta_ijk - T0), p)
                    m = m+1

#=======================================================================
#    if np.all(G == 0.0):
#        print(f'ERROR: found ZERO descriptor.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]}  NumEnvAt_a = {enviroment_atoms_a.shape[0]}')
#    if np.any(G == 0.0):
#        if (len(enviroment_atoms_a)  > 0) and (np.abs(np.abs(theta_ijk - T0) + np.pi) < 0.0000001):
#            print('-------------')
#            print(f'ERROR: found descriptor with ZERO coordinates.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]} NumEnvAt_a = {enviroment_atoms_a.shape[0]}: Rij={Rij} enviroment_atoms_a = {enviroment_atoms_a[:,-4:-1]}')
#            print(f'(abs(theta_ijk - T0) - Pi) = {np.abs(theta_ijk - T0) - np.pi}')
#            print(f'Descriptor G = {G}')
#            print('-------------')
#=======================================================================
    # print(f'=============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\nDistances:\n{enviroment_atoms_r[:,-1]}')
    # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\nDistances:\n{enviroment_atoms_a[:,-1]}')
    # print(f'Descriptors:\n{G}')
    return G


def generate_AEV_descriptors(waters, atoms):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """


    W = waters.shape[0]
    N_descriptors = len(generate_AEVs_X(waters[0], atoms))
    training_X = np.zeros([W, N_descriptors])
    #training_X = np.zeros([W, 7 * n])
    print(f'Number of descriptors per water molecule: {N_descriptors}')

    for i in range(W):
        training_X[i] = generate_AEVs_X(waters[i],atoms)
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        #if i > 2940: print(f' Descriptors for water {i}:\n{training_X[i]}')
    #exit()
    return training_X

def check_closest_env_distances(check_title,waters, atoms, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_P = 5.0
    cutoff_W = 3.5
    cutoff_clash = 2.1

    W = waters.shape[0]
    print(f'Checking positions of {check_title} with criteria:\ncutoff_P = {cutoff_P}\ncutoff_W = {cutoff_W}\ncutoff_clash = {cutoff_clash}')

    points_no_P = []
    points_no_W = []
    points_1_W = []
    points_clash = []
    idx_no_P = []
    idx_no_W = []
    idx_1_W = []
    idx_clash = []
    for i in range(W):
        closest_atoms_P     = atoms_within_cutoff(waters[i],                  atoms, cutoff_P)
        closest_atoms_W     = atoms_within_cutoff(waters[i], closest_atoms_P[:,:-1], cutoff_W)
        closest_atoms_clash = atoms_within_cutoff(waters[i], closest_atoms_W[:,:-1], cutoff_clash)
        if len(closest_atoms_P) == 0:
            points_no_P.append(waters[i,-3:])
            idx_no_P.append(i + pdb_idx_shift)
        if len(closest_atoms_W) == 0:
            points_no_W.append(waters[i,-3:])
            idx_no_W.append(i + pdb_idx_shift)
        if len(closest_atoms_W) == 1:
            points_1_W.append(waters[i,-3:])
            idx_1_W.append(i + pdb_idx_shift)
        if len(closest_atoms_clash) > 0:
            points_clash.append(waters[i,-3:])
            idx_clash.append(i + pdb_idx_shift)
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        #if i > 2940: print(f' Descriptors for water {i}:\n{training_X[i]}')

    #exit()
    title = check_title.replace(' ', '_')
    if len(points_no_P) > 0:
        nprint = len(points_no_P)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with noEnv within cutoff_P = {cutoff_P}')
        if nprint > 30: nprint = 30
        print(f'Indecies of first 30 sites with noEnv within cutoff_P = {cutoff_P}:', idx_no_P[:nprint])
        #print(f'points_no_P = {points_no_P[0:nprint]}')

        dump_pdb(np.array(points_no_P), f'Data Points {check_title} with noEnv, cut' + str(cutoff_P), title + '_noEnv_cut' + str(cutoff_P) + '.pdb', idx_no_P)

    if len(points_no_W) > 0:
        nprint = len(points_no_W)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with noEnv within cutoff_W = {cutoff_W}')
        if nprint > 30: nprint = 30
        print(f'Indecies of first 30 sites with noEnv within cutoff_W = {cutoff_W}:', idx_no_W[:nprint])
        #print(f'points_no_W = {points_no_W[0:nprint]}')

    if len(points_1_W) > 0:
        nprint = len(points_1_W)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with only 1 Env atom within cutoff_W = {cutoff_W}')
        print(f'These data points will have ZERO angular descriptor coords corresponding to tetha0=Pi because [1+cos(0 - Pi)] = 0.')
        if nprint > 30: nprint = 30
        print(f'Indecies of first 30 sites with only 1 Env atom within cutoff_W = {cutoff_W}:', idx_1_W[:nprint])
        #print(f'points_1_W = {points_1_W[0:nprint]}')

        dump_pdb(np.array(points_1_W), f'Data Points {check_title} with only 1 Env atom, cut' + str(cutoff_W), title + '_1Env_cut' + str(cutoff_W) + '.pdb', idx_1_W)
    if len(points_clash) > 0:
        nprint = len(points_clash)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with Env clash within cutoff_clash = {cutoff_clash}')
        if nprint > 30: nprint = 30
        print(f'Indecies of first 30 sites with Env clash within cutoff_clash = {cutoff_clash}:', idx_clash[:nprint])
        #print(f'points_clash = {points_clash[0:nprint]}')

        dump_pdb(np.array(points_clash),f'Data Points {check_title} with Env clash, cut' + str(cutoff_clash), title + '_clash_cut' + str(cutoff_clash) + '.pdb', idx_clash)
    #exit()

    return

def checkZERO_AEV_descriptors(siteTitle, training_X, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    siteTitle: string

    training_X: W x n x 7
    training X data
    ----------------------------------------------------------------------------
    Returns: NO
    """
    N_sites = training_X.shape[0]
    N_descriptors = training_X.shape[1]
    print(f'Checking {siteTitle} AEV descriptors: {N_sites} sites, descriptor dimension {N_descriptors}.')

    NumZeroDescriptors = 0
    NumZeroDescComponent = 0
    iZeroDescriptor=[]
    iZeroDescComponent=[]
    for i in range(N_sites):
        if np.all(training_X[i] == 0.0):
            NumZeroDescriptors = NumZeroDescriptors + 1
            iZeroDescriptor.append(i + pdb_idx_shift)
        if np.any(np.abs(training_X[i])  < 1e-16): # !!! np.any(training_X[i] == 0.0) IS NOT accurate for coordinbates X = 5e-38 due to rounding error
            NumZeroDescComponent = NumZeroDescComponent + 1
            iZeroDescComponent.append(i + pdb_idx_shift)

    #exit()
    if NumZeroDescriptors > 0:
        print('-------------')
        print(f'ERROR: For {siteTitle} found {NumZeroDescriptors} ZERO descriptors.')
        print(f'ERROR: ZERO AEV descriptor means that the site has no enviroment atoms within the cutoff.')
        print(f'ERROR: Make sure that the sites were correctly generated.')
        nprint = NumZeroDescriptors
        if nprint > 30: nprint = 30
        print('Indecies of first 30 sites with ZERO descriptor:', iZeroDescriptor[:nprint])


    if NumZeroDescComponent > 0:
        print('-------------')
        print(f'ERROR: Identified {NumZeroDescComponent} AEV descriptors with zero coordinates.')
        print(f'ERROR: ZERO AEV coordinate likely means that the site has 0 or only 1 enviroment atom within the cutoff.')
        print(f'ERROR: Make sure that this number equal to the Sum of Num of sites with 0 and only 1 enviroment atom.')
        print(f'ERROR: Otherwise, there might be some error. Make sure that the sites were correctly generated.')
        nprint = NumZeroDescComponent
        if nprint > 30: nprint = 30
        print('Indecies of first 30 sites with ZERO AEV coordinates:', iZeroDescComponent[:nprint])
        print('-------------')
    return

def generate_Z_descriptors(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """

    W = waters.shape[0]
    N_descriptors = 7 * n
    training_X = np.zeros([W, N_descriptors])
    print(f'Number of descriptors per water molecule: {N_descriptors}')
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()
    #exit()
    return training_X

def generate_AEV_descriptors_no_X(protein_atoms, water_atoms, cavities):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in xyz

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    total_atoms = np.append(protein_atoms, water_atoms, axis=0)
    cutoff_P = 3.3
    cutoff_W = 2.0
    # print("Partitioning protein atoms...")
    # atoms_partitions = get_input_partitions(atoms, partitions=2)
    # print("Checking atom count in partitions...")
    # is_same_count = check_num_of_protein_atoms(atoms_partitions, atoms)
    # if is_same_count:
    #     print("Partitioning successful")
    # else:
    #     print("Atom count error")
    #     exit()
    # num_of_partitions = len(atoms_partitions)
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    training_X = []
    cavgrids_no_P = []
    cavgrids_no_W = []
    cavgrids_near_W = []
    closest_at_dist = []
    for i in range(C):
        cav_point = cavities[i]
        #env_atoms = atoms_within_cutoff(cavities[i], atoms, 9.0)
        closest_atoms_W = atoms_within_cutoff(cav_point, water_atoms, cutoff_W)
        if len(closest_atoms_W) > 0:
            cavgrids_near_W.append(cav_point)
            continue
        
        closest_atoms_P = atoms_within_cutoff(cav_point, protein_atoms, cutoff_P)
        if len(closest_atoms_P) == 0:
            cavgrids_no_P.append(cav_point)
            continue
        ##
        ## NO Water cite is defined as
        ## Point with closest P atoms and no closest Water
        ##
        G = generate_AEVs_X(cav_point,  total_atoms)     # G = generate_AEVs_X(cav_point,atoms)
        training_X.append( G )

        #i_no = len(training_X)
        cavgrids_no_W.append(cav_point)

        # #HOH_check = closest_atoms_w[:, 2:4] - HOH_encoding
        # # print(f'cavities[i]: {cavities[i]}')
        # # print(f'near_atoms[:,:]:\n{near_atoms[:,:]}')
        # # print(f'closest_atoms[:,:]:\n{closest_atoms[:,:]}')
        # # print(f'HOH_check: {HOH_check}')
        # #n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        # #HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        # #if not np.any(HOH_check == 0.0):
        # if len(closest_atoms_W) == 0:
        #     # print(f'Identified as the NO water case')
        #     #G = generate_AEVs_X(cavities[i],  env_atoms[:,:-1])     # G = generate_AEVs_X(cavities[i],atoms)
        #     G = generate_AEVs_X(cav_point,  total_atoms)     # G = generate_AEVs_X(cav_point,atoms)
        #     training_X.append( G )
# 
        #     i_no = len(training_X)
        #     cavgrids_no_W.append(cav_point)
# 
        #     #closest_at_dist.append(n_nearest_atoms[0, -1])
        #     #print(f'cav_grid({i_no}) closest atom disdtance:{closest_at_dist[i_no - 1]}')
        #     #if  i_no <=5 :
        #     #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]} n_nearest_atoms[:, 0:4]:\n {n_nearest_atoms[:, 0:4]}')
        #     #    print(f'i_no={i_no} cav_grid({i}) = {cavities[i, 0:3]}:')
        #     #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]}:')
        #     #    #print(f'training_no_X[{i_no - 1}]:\n {training_X[i_no - 1]}')
        #     #    for a in range(n_nearest_atoms.shape[0]):
        #     #        #print(f'wat({i}) n_nearest_atoms[{a}]: {n_nearest_atoms[a, 0:4]} {internal_coords[a, :]} {n_nearest_atoms[a, -1]}')
        #     #        print(f'training_no_X[{i_no - 1}][{a}]: {training_X[i_no - 1][ a*7 : a*7 + 4]} {training_X[i_no - 1][ a*7+4 : a*7+7]} {n_nearest_atoms[a, -1]}')
        #     #else:
        #     #    exit()
        # else:
        #     cavgrids_near_W.append(cav_point)
        #exit()
    #for k in range(i_no):
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
    print(f'Number of generated cites with NO water (but near Protein): {len(training_X)}')
    print(f'Number of cavity grid points with NO Protein atoms within {cutoff_P}A: {len(cavgrids_no_P)}')
    print(f'Number of cavity grid points with water within {cutoff_W}A: {len(cavgrids_near_W)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')
    print(f'Last 5 descriptors: {training_X[-5:]}')
    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W','cavgrids_no_W.pdb')
    dump_pdb(np.array(cavgrids_no_P), 'cavgrids_no_P','cavgrids_no_P.pdb')
    dump_pdb(np.array(cavgrids_near_W), 'cavgrids_near_W','cavgrids_near_W.pdb')
    #exit()
    return np.array(training_X)


def noW_nearestN_cavity_grid(input_cavities, atoms, n):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in xyz

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    cavities: ndarray N x 3
    Array of cavity points in xyz that have no W within n nearest atoms
    """
    cavities = read_cavities(input_cavities)
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    cavgrids_no_W = []
    for i in range(C):
        cav_point = cavities[i]
        n_nearest_atoms = find_n_nearest_atoms(cav_point, atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            cavgrids_no_W.append(cav_point)

    print(f'Number of generated cites with NO water in {n} nearest atoms: {len(cavgrids_no_W)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')

    cav_name = os.path.basename(input_cavities).split('.')[0]
    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W', cav_name + '_no_W_nearest' + str(n) + '.pdb')
    #exit()
    return np.array(cavgrids_no_W)

def noW_cavity_grid(input_cavities, protein_atoms, water_atoms):
    """
    Filter out cavity grid points that are within cutoff_P and NOT within cutoff_W
    ----------------------------------------------------------------------------
    protein_atoms: ndarray N x 7
    Array of other atoms' information

    water_atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in xyz
    ----------------------------------------------------------------------------
    Returns:
    cavities: ndarray N x 3
    Array of cavity points in xyz that satisfy 2 criteria cutoff_P and NOT cutoff_W
    """
    cutoff_P = 3.3
    cutoff_W = 2.0

    cavities = read_cavities(input_cavities)
    C = cavities.shape[0]
    cavgrids_no_P = []
    cavgrids_no_W = []
    cavgrids_near_W = []
    for i in range(C):
        cav_point = cavities[i]
        #env_atoms = atoms_within_cutoff(cavities[i], atoms, 9.0)
        closest_atoms_W = atoms_within_cutoff(cav_point, water_atoms, cutoff_W)
        if len(closest_atoms_W) > 0:
            cavgrids_near_W.append(cav_point)
            continue
        
        closest_atoms_P = atoms_within_cutoff(cav_point, protein_atoms, cutoff_P)
        if len(closest_atoms_P) == 0:
            cavgrids_no_P.append(cav_point)
            continue
        ## NO Water cite is defined as
        ## Point with closest P atoms and no closest Water
        cavgrids_no_W.append(cav_point)

    cav_name = os.path.basename(input_cavities).split('.')[0]
    print(f'Number of generated cites with NO water (but near Protein): {len(cavgrids_no_W)}')
    print(f'Number of cavity grid points with water within {cutoff_W}A: {len(cavgrids_near_W)}')
    print(f'Number of cavity grid points with NO Protein atoms within {cutoff_P}A: {len(cavgrids_no_P)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')

    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W', cav_name + '_no_W_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    dump_pdb(np.array(cavgrids_near_W), 'cavgrids_near_W', cav_name + '_near_W_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    dump_pdb(np.array(cavgrids_no_P), 'cavgrids_no_P', cav_name + '_no_P_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    #exit()
    return np.array(cavgrids_no_W)
#===========================================================================
#============================= END of Modified =============================
#===========================================================================

def get_input_partitions(atoms, partitions=2):
    """
    calculate partitions for protein atoms; the size is defined by box_size
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of protein atoms' information

    box_size: float angstoms
    Size of a single partition
    ----------------------------------------------------------------------------
    Returns:
    atoms_partitions: ndarray num_of_boxes x atoms_in_box x 7
    """
    box_min = np.min(atoms[:, -3:], axis=0)
    box_max = np.max(atoms[:, -3:], axis=0)
    # box_length = box_max - box_min
    # num_of_boxes = (box_length / box_size).astype(int)
    partitions_x, step_x = np.linspace(box_min[0], box_max[0], partitions,
                                       retstep=True)
    partitions_y, step_y = np.linspace(box_min[1], box_max[1], partitions,
                                       retstep=True)
    partitions_z, step_z = np.linspace(box_min[2], box_max[2], partitions,
                                       retstep=True)
    atoms_partitions = []
    for one_x in partitions_x:
        for one_y in partitions_y:
            for one_z in partitions_z:
                atoms_within_box = find_atoms_within_box(one_x, step_x,
                                                         one_y, step_y,
                                                         one_z, step_z,
                                                         atoms)
                if atoms_within_box.any():
                    atoms_partitions.append(atoms[atoms_within_box])

    return atoms_partitions


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    W = waters.shape[0]
    training_X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()

    return training_X


def generate_training_yes_y(W):
    """
    generate y training data for yes cases
    ----------------------------------------------------------------------------
    W: int
    Number of water molecules or yes cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(W)
    zeros = np.zeros(W)
    training_y = np.append(ones[:, np.newaxis], zeros[:, np.newaxis], axis=1)

    return training_y


def check_num_of_protein_atoms(atoms_partitions, atoms):
    num_of_atoms_in_partitions = 0
    num_of_atoms = atoms.shape[0]
    for i in range(len(atoms_partitions)):
        one_P = atoms_partitions[i].shape[0]
        print(f"num of atoms in partition {i + 1}: {one_P}")
        num_of_atoms_in_partitions += one_P

    if num_of_atoms_in_partitions == num_of_atoms:
        return True

    print(f"there are {num_of_atoms_in_partitions} atoms in partitions")
    return False


def generate_training_no_X(atoms, cavities, n, interval: int):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in xyz

    n: int
    Number of closest atoms

    interval: int
    Interval between no cases
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    training_X = []
    closest_at_dist =[]
    for i in range(0, C, int(interval)):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            internal_coords = get_internal_coords(
                    n_nearest_atoms[:, -4:-1] - cavities[i, -3:])
            one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                       internal_coords, axis=1)
            training_X.append(one_training_X.flatten())
            #i_no = len(training_X)
            #closest_at_dist.append(n_nearest_atoms[0, -1])
            #print(f'cav_grid({i_no}) closest atom disdtance:{closest_at_dist[i_no - 1]}')
            #if  i_no <=5 :
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]} n_nearest_atoms[:, 0:4]:\n {n_nearest_atoms[:, 0:4]}')
            #    print(f'i_no={i_no} cav_grid({i}) = {cavities[i, 0:3]}:')
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]}:')
            #    #print(f'training_no_X[{i_no - 1}]:\n {training_X[i_no - 1]}')
            #    for a in range(n_nearest_atoms.shape[0]):
            #        #print(f'wat({i}) n_nearest_atoms[{a}]: {n_nearest_atoms[a, 0:4]} {internal_coords[a, :]} {n_nearest_atoms[a, -1]}')
            #        print(f'training_no_X[{i_no - 1}][{a}]: {training_X[i_no - 1][ a*7 : a*7 + 4]} {training_X[i_no - 1][ a*7+4 : a*7+7]} {n_nearest_atoms[a, -1]}')
            #else:
            #    exit()
    #for k in range(i_no):
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
        closest_at_dist.append(n_nearest_atoms[0, -1])
        if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of cavity NO-case Point-Atom(P,W) Distances', 2.3, 3.5)
    print(f'Number of generated sites with NO water: {len(training_X)}')
    return np.array(training_X)


def generate_training_no_y(P):
    """
    generate y training data for no cases
    ----------------------------------------------------------------------------
    P: int
    Number of protein atoms or no cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(P)
    zeros = np.zeros(P)
    training_y = np.append(zeros[:, np.newaxis], ones[:, np.newaxis], axis=1)

    return training_y

def add_rand_vector(atoms, length = 0.5):
    """
    add random vectors to atom positions
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of atoms' information
    length: length of all random vectors
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    random_vectors = np.random.rand(len(atoms), 3)
    magnitudes = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    unit_vectors = random_vectors / magnitudes
    pos_near_atoms = np.zeros([len(atoms), 7])
    pos_near_atoms[:,-3:] = length * unit_vectors
    pos_near_atoms = pos_near_atoms + atoms
    #print(f'atoms:{atoms[:2]}')
    #print(f'pos_near_atoms:{pos_near_atoms[:2]}')
    return pos_near_atoms


def draw_distance_histogram(values, nbins, Title, low_val_mark=2.4, high_val_mark=3.5):
    # Plotting a basic histogram
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(values, bins=nbins, color='skyblue', edgecolor='black')
    # Color the bars based on a condition
    for i, patch in enumerate(patches):
        if bins[i] < low_val_mark:
            patch.set_facecolor('red')
        elif bins[i] > high_val_mark:
            patch.set_facecolor('gray')
    # Adding labels and title
    plt.xlabel('Distance, (\u212B)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(Title, fontweight='bold')
    # Display the plot
    plt.show()

def generate_no_X_clash(check_title, waters, protein, cutoff_clash, n = 10, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    protein: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff = 4.5
    cutoff_HB = 3.5

    W = waters.shape[0]

    points_no_P = []
    points_clash_P = []
    idx_no_P = []
    idx_clash_P = []
    dist_P = []
    waters_low_E = []
    for i in range(W):
        W_i = waters[i]
        closest_atoms_P = atoms_within_cutoff(W_i, protein, cutoff)
        closest_atoms_clash_P = atoms_within_cutoff(W_i,
                                                    closest_atoms_P[:, :-1],
                                                    cutoff_clash)

        if len(closest_atoms_P) == 0:
            points_no_P.append(waters[i,-3:])
            idx_no_P.append(i + pdb_idx_shift)
        else:
            closest_atoms_P = closest_atoms_P[np.argsort(closest_atoms_P[:,-1])] # SORT BY DISTANCE idx=[-1]
            dist_P.append(closest_atoms_P[0,-1]) # idx=0 closest atom because closest_atoms is ordered array

        if len(closest_atoms_clash_P) > 0:
            points_clash_P.append(waters[i,-3:])
            idx_clash_P.append(i + pdb_idx_shift)
        else:
            waters_low_E.append(waters[i])

    if len(points_clash_P) > 0:
        nprint = len(points_clash_P)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with P-Env clash within cutoff_clash = {cutoff_clash}')
        print(f'Will use these {nprint} positions of clashed water as NO cases')
        print(f'Number of remained water Yes cases after excluding {nprint} positions of clashed water is {len(waters_low_E)}.')
        if nprint > 30: nprint = 30
        print(f'Indecies of first 30 sites with P-Env clash within cutoff_clash = {cutoff_clash}:', idx_clash_P[:nprint])

    print(f'Computed distnaces for {len(dist_P)} water molecules and closest Protein atom within cutoff')
    print(f'{dist_P[0:5]}')
    dist_sorted = np.sort(dist_P)
    print(f'{dist_sorted[0:10]}')

    draw_distance_histogram(dist_P, 30, 'Distribution of Protein-Water Distances', cutoff_clash, cutoff_HB)
    #exit()
    atoms= np.append(waters, protein, axis=0)
    training_X = []
    for i in range(len(points_clash_P)):
        n_nearest_atoms = find_n_nearest_atoms(points_clash_P[i], atoms, n)
        internal_coords = get_internal_coords(n_nearest_atoms[:, -4:-1] - points_clash_P[i])
        one_training_X = np.append(n_nearest_atoms[:, 0:4], internal_coords, axis=1)
        training_X.append(one_training_X.flatten())

    print(f'Number of generated sites with NO water: {len(training_X)}')
    return np.array(training_X), np.array(waters_low_E)


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    W = waters.shape[0]
    training_X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()

    return training_X



def feature_encoder_atom(feature_number):
    """
    takes a number and outputs [cos(number), sin(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents atom_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [cos(number), sin(number)]

    """
    return np.array([np.cos(feature_number), np.sin(feature_number)])


def feature_encoder_residue(feature_number):
    """
    takes a number and outputs [sin(number), cos(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents residue_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [sin(number), cos(number)]

    """
    return np.array([np.sin(feature_number), np.cos(feature_number)])


def read_pdb(input_pdb):
    """
    reads a pdb file and returns numpy array of water data and protein data
    ----------------------------------------------------------------------------
    input_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    water_data, protein_data: ndarray: N x 7
    """
    # read in the pdb file
    pdb_file = open(input_pdb)
    atom_info = [line for line in pdb_file.readlines()
                 if line.startswith('ATOM  ') or line.startswith('HETATM')]
    water_data = []
    protein_data = []
    num_of_atom_types = len(atom_types.keys())
    num_of_residue_types = len(residue_types.keys())
    for line in atom_info:
        one_data = np.array([])
        xyz = [float(x) for x in line[30:53].split()]
        # read in the atom name
        atom_type = str(line[13:16]).strip()
        res_type = str(line[17:20]).strip()
        try:
            atom_encode = feature_encoder_atom(atom_types[atom_type])
        except KeyError:
            num_of_atom_types += 1
            atom_types[atom_type] = num_of_atom_types
            atom_encode = feature_encoder_atom(atom_types[atom_type])
            # print("atom_types:", atom_types)
        try:
            residue_encode = feature_encoder_residue(residue_types[res_type])
        except KeyError:
            num_of_residue_types += 1
            residue_types[res_type] = num_of_residue_types
            residue_encode = feature_encoder_residue(residue_types[res_type])
            # print("residue_types:", residue_types)

        one_data = np.append(one_data, atom_encode)
        one_data = np.append(one_data, residue_encode)
        one_data = np.append(one_data, xyz)
        if res_type == 'HOH':
            water_data.append(one_data)
        else:
            protein_data.append(one_data)

    return np.array(water_data), np.array(protein_data)


def read_cavities(cavities_pdb):
    """
    reads a pdb file and returns numpy array of cavity data
    ----------------------------------------------------------------------------
    cavities_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    cavities_data: ndarray: N x 3
    """
    # read in the pdb file
    pdb_file = open(cavities_pdb)
    cav_info = [line for line in pdb_file.readlines() if
                line.startswith('HETATM')]
    cavities_data = []
    for line in cav_info:
        xyz = [float(x) for x in line[30:53].split()]
        cavities_data.append(xyz)

    return np.array(cavities_data)

def combine_training_data(X_yes, X_no, y_yes, y_no):
    num_no_cases = y_no.shape[0]
    num_yes_cases = y_yes.shape[0]
    training_X = np.append(X_no, X_yes, axis=0)
    training_y = np.append(y_no, y_yes, axis=0)
    ratio = int(num_no_cases / num_yes_cases)
    yes_i = num_no_cases
    for i in range(num_yes_cases):
        training_X[[i + ratio, yes_i + i]] = \
            training_X[[yes_i + i, i + ratio]]
        training_y[[i + ratio, yes_i + i]] = training_y[[yes_i + i, i + ratio]]

    return training_X, training_y

def randomize_training_data(training_X, training_y):
    assert training_X.shape[0] == training_y.shape[0]
    p = np.random.permutation(training_X.shape[0])
    return training_X[p], training_y[p]

def print_arr_nByRow(arr, nByRow = 7, nprec=4):
    arr = np.array(arr)
    for i in range(0, len(arr), nByRow):
        formatted_str = "%.*f" %  (nprec, arr.item(i))
        for x in arr[i + 1:i + nByRow]:
           formatted_str = "%s %.*f" % (formatted_str, nprec,x)
        print(formatted_str)

def check_conserved_components(arr2d,arrname='arr2d'):
    # Reduce along columns (axis=0)
    min_val = np.min(arr2d, axis=0)
    max_val = np.max(arr2d, axis=0)
    delta = max_val - min_val
    if np.any(delta == 0.0):
        print(f'There are conserved coordinates in the array "{arrname}".')
        print(f'delta = max_val - min_val:')
        print_arr_nByRow(delta, 7, 8)
        zero_indices = np.where(delta == 0)[0]
        print(f'Indecies of conserved coordinates in the array "{arrname}":{zero_indices}')


if __name__ == '__main__':
    try:
        input_pdb = sys.argv[1]
        input_cavities = sys.argv[2]
    except IndexError:
        print("Usage: python pdb_input_processing.py input_pdb input_cavities")
        exit()
    pdb_name = os.path.basename(input_pdb).split('.')[0]
    water_data, protein_data = read_pdb(input_pdb)
    cavities_data = read_cavities(input_cavities)
    # print(atom_types)
    total_data = np.append(water_data, protein_data, axis=0)
    print("Generating training data...")
    starting_time = timeit.default_timer()

    ##
    ## Generate Z Descriptors (default)
    ##
    training_no_X_clash, water_OK = generate_no_X_clash("water", water_data,
                                                        protein_data, 2.3)
    site_near_protein = add_rand_vector(protein_data, 0.5)
    training_no_X_prot = generate_training_no_X(total_data, site_near_protein,
                                                n=10, interval=10)

    training_yes_X = generate_training_yes_X(water_OK, total_data, n=10)
    num_of_cav = cavities_data.shape[0]
    print("number of no cases before balancing: %d" % num_of_cav)
    interval_of_no_cases = 1  # int(num_of_cav / water_data.shape[0])
    #interval_of_no_cases = int(num_of_cav / training_yes_X.shape[0])
    training_no_X = generate_training_no_X(total_data, cavities_data, n=10,
                                           interval=interval_of_no_cases)
    print("number of yes cases: %d" % training_yes_X.shape[0])
    print("number of no cases: %d" % training_no_X.shape[0])

    #training_yes_X = generate_Z_descriptors(water_data, total_data, n=10)
    #noW_cav_sites = noW_nearestN_cavity_grid(input_cavities, total_data, n=10)
    #training_no_X = generate_Z_descriptors(noW_cav_sites, total_data, n=10)



    ##   ##
    ##   ## Generate AEV Descriptors
    ##   ##
    ##   check_closest_env_distances('Water', water_data, total_data, len(protein_data))
    ##   noW_cav_sites = noW_nearestN_cavity_grid(input_cavities, total_data, n=10)
    ##   #noW_cav_sites = noW_cavity_grid(input_cavities, protein_data, water_data)
    ##   check_closest_env_distances('CavGrid', noW_cav_sites, total_data, 0)
    ##   # Generate AEV Descriptors
    ##   training_yes_X = generate_AEV_descriptors(water_data, total_data)
    ##   checkZERO_AEV_descriptors('PDB Water', training_yes_X, len(protein_data))
    ##   training_no_X = generate_AEV_descriptors(noW_cav_sites, total_data)
    ##   checkZERO_AEV_descriptors('noW Cavity Grid', training_no_X)

    training_yes_y = generate_training_yes_y(water_OK.shape[0])
    training_no_y = generate_training_no_y(training_no_X.shape[0])
    #exit()

    # training_X, training_y = combine_training_data(training_yes_X,
    #                                                training_no_X,
    #                                                training_yes_y,
    #                                                training_no_y)
    training_X = np.append(training_yes_X, training_no_X, axis=0)
    training_y = np.append(training_yes_y, training_no_y, axis=0)

    # Add clashed water to NO cases
    training_no_y_clash = generate_training_no_y(training_no_X_clash.shape[0])
    training_X = np.append(training_X, training_no_X_clash, axis=0)
    training_y = np.append(training_y, training_no_y_clash, axis=0)
    # Add protein atom positions as water NO cases
    print('Generate proten NO cases y values')
    training_no_y_prot = generate_training_no_y(training_no_X_prot.shape[0])
    print('Finished proten NO cases y values')
    training_X = np.append(training_X, training_no_X_prot, axis=0)
    training_y = np.append(training_y, training_no_y_prot, axis=0)
    # training_X, training_y = combine_training_data(training_X,
    #                                                training_no_X_clash,
    #                                                training_y,
    #                                                training_no_y_clash)

    ##
    ##  Check conserved components of descriptors
    ##
    # check_conserved_components(training_yes_X,arrname='training_yes_X')
    # check_conserved_components(training_no_X,arrname='training_no_X')
    # check_conserved_components(training_no_X_clash,arrname='training_no_X_clash')
    # check_conserved_components(training_no_X_prot,arrname='training_no_X_prot')

    ending_time = timeit.default_timer()
    total_time = ending_time - starting_time
    print(f"Data processing took {total_time:.2f} seconds")
    training_X, training_y = randomize_training_data(training_X, training_y)
    np.save(f'train_data/{pdb_name}_CI_X_yes.npy', training_yes_X)
    np.save(f'train_data/{pdb_name}_CI_y_yes.npy', training_yes_y)
    np.save(f'train_data/{pdb_name}_CI_X.npy', training_X)
    np.save(f'train_data/{pdb_name}_CI_y.npy', training_y)
    # My extra output for testing
    np.save(f'train_data/{pdb_name}_CI_X_no.npy', training_no_X)
    np.save(f'train_data/{pdb_name}_CI_y_no.npy', training_no_y)
    np.save(f'train_data/{pdb_name}_CI_X_no_clash.npy', training_no_X_clash)
    np.save(f'train_data/{pdb_name}_CI_y_no_clash.npy', training_no_y_clash)
    np.save(f'train_data/{pdb_name}_CI_X_no_prot.npy', training_no_X_prot)
    np.save(f'train_data/{pdb_name}_CI_y_no_prot.npy', training_no_y_prot)

    max_yes = np.max(training_yes_X)
    min_yes = np.min(training_yes_X)
    max_no = np.max(training_no_X)
    min_no = np.min(training_no_X)
    print(f'max and min of yes descriptors: {max_yes:.3f}, {min_yes:.3f}')
    print(f'max and min of no descriptors: {max_no:.3f}, {min_no:.3f}')
    print(f'Last 5 descriptors: {training_no_X[-5:]}')
    print(f'Number of generated water sites: {len(training_yes_X)}')
    print(f'Number of generated NO water sites: {len(training_no_X)}')
    print(f'Number of generated clash NO water sites: {len(training_no_X_clash)}')
    print(f'Number of generated protein atom NO water sites: {len(training_no_X_prot)}')
    print(f'Total Number of generated data points: {len(training_X)}')
    pass
