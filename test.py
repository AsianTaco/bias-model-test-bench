from bias_bench.io import BiasModelData
from bias_bench.analysis.two_point import plot_power_spectrum
from bias_bench.analysis.one_point import plot_one_point_stats
from bias_bench.analysis.three_point import plot_bispectrum
from bias_bench.benchmark_models import TruncatedPowerLaw
from bias_bench.Params import BiasParams
import matplotlib.pyplot as plt

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
    plt.style.use(f"./plot_styles/{params['plotting_style']}")

    # Make plots.
    if 'ngal_vs_rho' in params['plots']:
        plot_one_point_stats(BM)
    if 'power_spectrum' in params['plots']:
        plot_power_spectrum(BM, params)

    #plot_power_spectrum(BM, pylians=True)
    #plot_bispectrum(BM)
