import h5py
import numpy as np


class BiasModelData:

    def __init__(self, params):

        # Parameters from parameter file.
        self.params = params

        # Load the data.
        self._load_bias_model_data()

    def _load_bias_model_data(self):
        """ Load data from the HDF5 file """

        self.info = {}
        _required_header_params = ['BoxSize', 'GridSize']

        with h5py.File(self.params['hdf5_file_path'], "r") as f:
            # Read the meta-data from the header
            for att in f['Header'].attrs.keys():
                self.info[att] = f['Header'].attrs.get(att)

            # Make sure we have the minimum ammount.
            for att in _required_header_params:
                assert att in self.info.keys()

            # The over density field (NxNxN)
            # TODO: consistent naming
            self.overdensity_field = f[self.params['overdensity_field_name']][...]
            print(f"Loaded overdensity field shape={self.overdensity_field.shape}")
            assert np.all(self.overdensity_field.shape == self.info['GridSize'])

            # Count field (predicted)
            if self.params['count_field_name'] in f:
                self.count_field = f[self.params['count_field_name']][...]
                print(f"Loaded predicted count field shape={self.count_field.shape}")
                assert self.overdensity_field.shape == self.count_field.shape

            # Count field (truth)
            if self.params['count_field_truth_name'] in f:
                self.count_field_truth = f[self.params['count_field_truth_name']][...]
                print(f"Loaded count field truth shape={self.count_field_truth.shape}")
                assert self.overdensity_field.shape == self.count_field_truth.shape

