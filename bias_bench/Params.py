import yaml

DefaultParameter = {
    "plotting_style": "nature.mplstyle",
    "predict_counts": None,
    "power_spectrum": {'show_density': False, 'MAS': None},
    "bi_spectrum": {'show_density': False, 'k1': 0.5, 'k2': 0.6, 'Ntheta': 25, 'MAS': None}
}

DefaultSubParameter = {
    "overdensity_field_name": "overdensity_field",
    "count_field_truth_name": "count_field_truth",
    "count_field_name": "count_field",
    "count_field_benchmark_name": "truncated power law",
}

class BiasParams:
    """
    Load parameters from the parameter YAML file.

    If a given parameter is not present in the parameter file, a default will
    be used.

    Some parameters are required, see below.

    Attributes
    ----------
    data : dict
        Dictionary storing the bias_bench parameters

    Required parameters in YAML file
    --------------------------------
    hdf5_file_path : string
        Path to HDF5 file that contains the fields

    plots : list of strings
        The plots to make, options are:
            - power_spectrum
            - ngal_vs_rho

    Optional IO parameters in YAML file
    -----------------------------------
    overdensity_field_name : string
        HDF5 dataset name containing the dark matter overdensity field
        e.g, "refl0025n0376/ngrid32/density" for eagle.hdf5
    count_field_truth_name : string
        HDF5 dataset name containing the ground truth count field
    count_field_name : string
        HDF5 dataset name containing the biad model predicted count field

    Optional power spectrum plot options in YAML file
    -------------------------------------------------
    kmin : float
        Minimum k (Mpc/h)
    kmax : float
        Maximum k (Mpc/h)
    Nk : float
        Num of k bins between min and max
    """

    def __init__(self, param_file):

        self.param_file = param_file
        self.data = self._load_params_from_yaml()

        self._append_default_values()

        self._sanity_checks()
        self._print_params()

    def _load_params_from_yaml(self):

        with open(self.param_file) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # Check we have required params.
        _required = ['plots', 'num_bias_models', 'bias_model_1']
        for att in _required:
            assert att in data.keys(), f"Need {att} as param"

        # Check we have required sub-params.
        _required = ['hdf5_file_path']
        for i in range(data['num_bias_models']):
            for att in _required:
                assert att in data[f'bias_model_{i+1}'].keys(), f"Need {att} as param"
        
        return data

    def _print_params(self):
        """ Print out parameters to terminal. """
        OKGREEN = "\033[92m"
        OKCYAN = "\033[96m"
        ENDC = "\033[0m"

        print(f"----------")
        print(f"Loaded parameter file {self.param_file}")
        print(f"----------")
        for att in self.data:
            print(f"{OKGREEN}{att}{ENDC}: {OKCYAN}{self.data[att]}{ENDC}")
        print(f"----------")

    def _append_default_values(self):

        # Default overall parameters
        for att in DefaultParameter.keys():
            if att not in self.data.keys():
                self.data[att] = DefaultParameter[att]

        # Default sub parameters
        for i in range(self.data['num_bias_models']):
            for att in DefaultSubParameter.keys():
                if att not in self.data[f'bias_model_{i+1}'].keys():
                    self.data[f'bias_model_{i+1}'][att] = DefaultSubParameter[att]

    def _sanity_checks(self):
        # Make sure power spectrum params are correct
        for att in ['show_density', 'MAS']:
            assert att in self.data['power_spectrum'].keys(), f"Missing {att}"

        for att in ['show_density', 'k1', 'k2', 'Ntheta', 'MAS']:
            assert att in self.data['bi_spectrum'].keys(), f"Missing {att}"
