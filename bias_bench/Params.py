import yaml

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

    Optional IO parameters in YAML file
    -----------------------------------
    overdensity_field_name : string
        HDF5 dataset name containing the dark matter overdensity field
        e.g, "refl0025n0376/ngrid32/density" for eagle.hdf5
    count_field_truth_name : string
        HDF5 dataset name containing the ground truth count field
    count_field_name : string
        HDF5 dataset name containing the biad model predicted count field


    """

    def __init__(self, param_file):

        self.param_file = param_file

        # Load parameters from yaml file.
        self._load_params()

        # Append default values.
        self._mix_with_defaults()

        # Print params.
        self._print_params()

    def _load_params(self):

        with open(self.param_file) as file:
            self.data = yaml.load(file, Loader=yaml.FullLoader)

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

    def _mix_with_defaults(self):

        _defaults = {
            "overdensity_field_name": "overdensity_field",
            "count_field_truth_name": "count_field_truth",
            "count_field_name": "count_field",
            "plotting_style": "nature.mplstyle",
            "predict_counts": None,
            "power_spectrum": None
        }

        for att in _defaults.keys():
            if att not in self.data.keys():
                self.data[att] = _defaults[att]
