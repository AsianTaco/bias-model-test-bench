from bias_bench.io import BiasModelData
from bias_bench.analysis.two_point import plot_power_spectrum
from bias_bench.analysis.one_point import plot_one_point_stats
from bias_bench.analysis.three_point import plot_bispectrum
from bias_bench.benchmark_models import TruncatedPowerLaw
import matplotlib.pyplot as plt

import yaml


class BiasParams:

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
            "predict_counts": None
        }

        for att in _defaults.keys():
            if att not in self.data.keys():
                self.data[att] = _defaults[att]


def _predict_galaxy_counts(BM, params):
    """ Predict ngal from density field. """

    _allowed_benchmark_models = ['truncated_power_law']
    assert params['predict_counts_model'] in _allowed_benchmark_models

    # Flatten to 1D arrays.
    delta_flattened = BM.overdensity_field.flatten()
    counts_flattened = BM.count_field_truth.flatten()

    if params['predict_counts_model'] == "truncated_power_law":
        pl_model = TruncatedPowerLaw()

        # Fit ngal vs delta relation.
        fitted_params = pl_model.fit(delta_flattened, counts_flattened)

        # Get mean ngal for a given value of delta.
        count_mean = pl_model.predict(delta_flattened, fitted_params)

        # Poisson sample a realization using the mean value as the Poisson mu.
        predicted_count_field = pl_model.sample(delta_flattened, fitted_params).reshape(BM.count_field_truth.shape)

    # Store predicted count field.
    BM.count_field = predicted_count_field


if __name__ == '__main__':

    # Load information from parameter file.
    params = BiasParams("eagle_25.yml")
    params = params.data

    # Load data.
    BM = BiasModelData(params)

    # Predict counts using benchmark models.
    if params["predict_counts_model"] is not None:
        _predict_galaxy_counts(BM, params)

    # Set plotting style.
    plt.style.use("nature.mplstyle")

    # Make plots.
    plot_one_point_stats(BM)
    plot_power_spectrum(BM)
    plot_bispectrum(BM)
