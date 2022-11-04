import h5py
import numpy as np


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


    def __init__(self, params):

        # Parameters from parameter file.
        self.params = params

        # Load the data.
        self._load_bias_model_data()

    def _load_bias_model_data(self):
        """ Load data from the HDF5 file """

        self.info = {}

        with h5py.File(self.params['hdf5_file_path'], "r") as f:

            # The over density field.
            self.overdensity_field = f[self.params['overdensity_field_name']][...]
            print(f"Loaded overdensity field shape={self.overdensity_field.shape}")
            
            # Load simulation info from attributes of the overdensity dataset.
            for att in f[self.params['overdensity_field_name']].attrs.keys():
                self.info[att] = f[self.params['overdensity_field_name']].attrs.get(att)
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

        # Make sure we have the minimum needed simulation information.
        _required_header_params = ['BoxSize', 'GridSize']
        for att in _required_header_params:
            assert att in self.info.keys()

        # Sanity checks.
        assert self.info["BoxSize"] > 0.0
        for att in ["overdensity_field", "count_field", "count_field_truth"]:
            if hasattr(self, att):
                assert np.all(getattr(self, att).shape == self.info['GridSize'])
