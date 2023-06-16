import h5py

from bias_bench.Params import BiasParams
from bias_bench.constants import *
from bias_bench.utils import bias_bench_print


class BiasModelData:
    """
    Loads the bias model data from the HDF5 file.

    Stores the dark matter density field, predicted count field and true count
    field.

    Information about the simulation are expected to be stored as attributes in
    the density field dataset (minimum required "BoxSize" and "GridSize").

    Parameters
    ----------
    params : dict
        Stores the bias_bench run parameters
    model_index : int
        Which bias model in the parameter file are we loading

    Attributes
    ----------
    overdensity_field : 3d array [Ngrid,Ngrid,Ngrid]
        The dark matter overdensity field
    count_field : 3d array [Ngrid,Ngrid,Ngrid]
        Galaxy or subhalo counts field (predicted from bias model)
    count_field_truth : 3d array [Ngrid,Ngrid,Ngrid]
        Galaxy or subhalo counts field (underlying truth from simulation)

    info : dict
        Stores parameters about the simulation
    """

    def __init__(self, params: BiasParams, model_index=1):

        # Parameters from parameter file.
        self.params = params.data
        self.which_model = model_index
        self.dm_overdensity_fields = []
        self.count_fields_truth = []
        self.count_fields_predicted = []
        self.counts_field_benchmark = []
        self.info = {}

        # Load the data.
        # TODO: Add option to not load anything from data
        # TODO: Add option to load a specific range of res, mass, bins, simulations
        self.bias_model_params = self.params[f'bias_model_{self.which_model}']
        self.n_simulations = self.bias_model_params['n_simulations']
        self.n_res = self.bias_model_params['n_res']
        self.n_mass_bins = self.bias_model_params['n_mass_bins']

        self._load_and_set_bias_model_data_from_file()

    def _load_and_set_bias_model_data_from_file(self):
        """ Load data from the HDF5 file """

        sim_group_name = self.bias_model_params['simulation_group_name']

        n_overdensity_fields = 0
        n_counts_fields = 0
        n_counts_pred_fields = 0

        with h5py.File(self.bias_model_params['hdf5_file_path'], "r") as f:

            # Loading the overdensity fields at different resolutions
            for sim_i in range(self.n_simulations):
                overdensity_fields_per_resolution = []
                counts_field_per_res = []
                counts_field_predicted_per_res = []

                for res_i in range(self.n_res):

                    # Extract overdensity field with different resolutions
                    res_group_name = f'{sim_group_name}_{sim_i}/{res_base_name}_{res_i}'
                    overdensity = f[f'{res_group_name}/{dm_overdensity_name}'][...]
                    bias_bench_print(
                        f'Loaded overdensity field from {sim_group_name}_{sim_i} with shape={overdensity.shape}')
                    overdensity_fields_per_resolution.append(overdensity)
                    n_overdensity_fields += 1

                    # FIXME: Think about if box and mass bin configurations across different simulation are allowed
                    #  to differ

                    if sim_i == 0:
                        self.info[f'{res_base_name}_{res_i}'] = {}
                        self.info[f'{res_base_name}_{res_i}'][box_size_attr] = f[res_group_name].attrs.get(
                            box_size_attr)
                        self.info[f'{res_base_name}_{res_i}'][n_grid_attr] = f[res_group_name].attrs.get(n_grid_attr)

                    # Extract ground truth and predicted counts field for different mass bins
                    counts_field_per_res_and_mass_bin = []
                    counts_field_pred_per_res_and_mass_bin = []
                    for mass_bin_i in range(self.n_mass_bins):
                        counts_field_dataset_name = f'{res_group_name}/{counts_field_truth_base_name}_{mass_bin_i}'
                        if counts_field_dataset_name in f:

                            if sim_i == 0:
                                lo = f[counts_field_dataset_name].attrs.get(mass_bin_left_attr)
                                hi = f[counts_field_dataset_name].attrs.get(mass_bin_right_attr)
                                self.info[f'{res_base_name}_{res_i}'][f'mass_bin_{mass_bin_i}'] = [lo, hi]

                            counts_field = f[counts_field_dataset_name][...]
                            counts_field_per_res_and_mass_bin.append(counts_field)
                            n_counts_fields += 1

                        counts_field_pred_dataset_name = f'{res_group_name}/{counts_field_predicted_base_name}_{mass_bin_i}'
                        if counts_field_pred_dataset_name in f:

                            if sim_i == 0:
                                lo = f[counts_field_pred_dataset_name].attrs.get(mass_bin_left_attr)
                                hi = f[counts_field_pred_dataset_name].attrs.get(mass_bin_right_attr)
                                self.info[f'{res_base_name}_{res_i}'][f'mass_bin_{mass_bin_i}'] = [lo, hi]

                            counts_field = f[counts_field_pred_dataset_name][...]
                            counts_field_pred_per_res_and_mass_bin.append(counts_field)
                            n_counts_pred_fields += 1

                        # TODO: Add possibility to read in benchmark model file
                        # counts_field_pred_dataset_name = f'{res_group_name}/{counts_field_predicted_base_name}_{mass_bin_i}'
                        # if counts_field_pred_dataset_name in f:
                        #
                        #     if sim_i == 0:
                        #         lo = f[counts_field_pred_dataset_name].attrs.get(mass_bin_left_attr)
                        #         hi = f[counts_field_pred_dataset_name].attrs.get(mass_bin_right_attr)
                        #         self.info[f'{res_base_name}_{res_i}'][f'mass_bin_{mass_bin_i}'] = [lo, hi]
                        #
                        #     counts_field = f[counts_field_pred_dataset_name][...]
                        #     counts_field_pred_per_res_and_mass_bin.append(counts_field)
                        #     n_counts_pred_fields += 1

                    counts_field_per_res.append(counts_field_per_res_and_mass_bin)
                    counts_field_predicted_per_res.append(counts_field_pred_per_res_and_mass_bin)

                self.dm_overdensity_fields.append(overdensity_fields_per_resolution)
                self.count_fields_truth.append(counts_field_per_res)
                self.count_fields_predicted.append(counts_field_predicted_per_res)

        print(f"Loaded {n_overdensity_fields} overdensity fields.")
        print(f"Loaded {n_counts_fields} ground truth counts fields.")
        print(f"Loaded {n_counts_pred_fields} predicted counts fields.")

        # Make sure we have the minimum needed simulation information.
        _required_header_params = [box_size_attr, n_grid_attr]
        for att in _required_header_params:
            assert att in self.info['res_0'].keys()

        # Sanity checks.
        # TODO: Re-activate sanity checks
        # assert self.info["BoxSize"] > 0.0
        # for att in ["overdensity_field", "count_field", "count_field_truth"]:
        #     if hasattr(self, att):
        #         assert np.all(getattr(self, att).shape == self.info['GridSize'])
