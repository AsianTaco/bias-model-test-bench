import numpy as np
import h5py

from bias_bench.constants import *


def bias_bench_print(txt, verbose=False):
    if verbose:
        print(txt)


def add_count_fields_to_hdf5(file_path: str, group_name: str, count_fields: np.ndarray, mass_bins,
                             is_ground_truth: bool, all_mass_bin: bool):
    if is_ground_truth:
        data_set_base_name = f'{group_name}/{counts_field_truth_base_name}'
    else:
        data_set_base_name = f'{group_name}/{counts_field_predicted_base_name}'

    with h5py.File(file_path, "r+") as f:
        for i in range(len(mass_bins) - 1):
            dset = f.create_dataset(f"{data_set_base_name}_{i}", data=count_fields[i])

            dset.attrs.create(mass_bin_left_attr, mass_bins[i])
            dset.attrs.create(mass_bin_right_attr, mass_bins[i + 1])

        if all_mass_bin:
            dset = f.create_dataset(f"{data_set_base_name}_{len(mass_bins)}", data=count_fields[-1])

            dset.attrs.create(mass_bin_left_attr, mass_bins[0])
            dset.attrs.create(mass_bin_right_attr, np.inf)


def add_overdensity_field_to_hdf5(file_path: str, group_name: str, dm_overdensity_fields: np.array):
    with h5py.File(file_path, "r+") as f:
        for res_i, dm_overdensity_field in enumerate(dm_overdensity_fields):
            f.create_dataset(f"{group_name}/{res_base_name}_{res_i}/{dm_overdensity_name}", data=dm_overdensity_field)


def create_hdf5_data_file(file_path: str, parent_group_name: str, n_sims, n_res, boxsize, ngrid):
    with h5py.File(file_path, 'r+') as f:
        for sim_i in range(n_sims):
            f.create_group(f"{parent_group_name}_{sim_i}")
            for res_i in range(n_res):
                res_group_name = f"{parent_group_name}_{sim_i}/{res_base_name}_{res_i}"
                f.create_group(res_group_name)
                f[res_group_name].attrs.create(box_size_attr, boxsize)
                f[res_group_name].attrs.create(n_grid_attr, ngrid)
