import numpy as np
import h5py

from bias_bench.constants import *


def bias_bench_print(txt, verbose=False):
    if verbose:
        print(txt)


def add_count_fields_to_hdf5(file_path: str, group_name: str, count_fields: np.ndarray, mass_bins, is_ground_truth: bool):
    if is_ground_truth:
        data_set_base_name = f'{group_name}/{counts_field_truth_base_name}'
    else:
        data_set_base_name = f'{group_name}/{counts_field_predicted_base_name}'

    with h5py.File(file_path, "r+") as f:
        for i in range(len(mass_bins) - 1):
            dset = f.create_dataset(f"{data_set_base_name}_{i}", data=count_fields[i])

            dset.attrs.create(mass_bin_left_attr, mass_bins[i])
            dset.attrs.create(mass_bin_right_attr, mass_bins[i + 1])
