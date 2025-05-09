import numpy as np

from bias_bench.constants import *


def get_halo_coords_per_mass_bin(m_bins, cat):
    return [cat[[x_key, y_key, z_key]][
                np.where((m_bins[i] <= cat[mvir_key]) & (cat[mvir_key] < m_bins[i + 1]))] for
            i in range(len(m_bins) - 1)]


def get_halo_count_field(halo_coords, mesh_size, box_size):
    if halo_coords.size == 0:
        return np.zeros((mesh_size, mesh_size, mesh_size))
    else:
        x, y, z = halo_coords[x_key], halo_coords[y_key], halo_coords[z_key]
        box_bounds = [(0, box_size)] * 3
        n_halos_per_voxel, _ = np.histogramdd(np.stack([x, y, z], axis=1), bins=mesh_size, range=box_bounds)
        return n_halos_per_voxel
