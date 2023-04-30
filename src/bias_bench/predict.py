import numpy as np

from bias_bench.data_io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.benchmark_models.TruncatedPowerLaw import TruncatedPowerLaw
from bias_bench.utils import bias_bench_print


def predict_galaxy_counts(bias_model_data: BiasModelData, bias_params: BiasParams,
                          which_model=1):
    """ Predict ngal from density field. """

    params = bias_params.data

    _allowed_benchmark_models = ['truncated_power_law']
    benchmark_model_name = params[f'bias_model_{which_model}']['predict_counts_model']
    assert benchmark_model_name in _allowed_benchmark_models

    counts_field_benchmark = [
        [
            [
                [] for _ in range(bias_model_data.n_mass_bins)
            ] for _ in range(bias_model_data.n_res)
        ] for _ in range(bias_model_data.n_simulations)
    ]

    n_benchmark_fits = 0

    # TODO: make this more flexible
    benchmark_model = TruncatedPowerLaw()

    for res_i in range(bias_model_data.n_res):
        for mass_bin_i in range(bias_model_data.n_mass_bins):
            overdensities = []
            counts_fields_truth = []

            for sim_i in range(bias_model_data.n_simulations):
                overdensities.append(bias_model_data.dm_overdensity_fields[sim_i][res_i])

                try:
                    counts_fields_truth.append(bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i])
                except IndexError:
                    print(f"No truth count field found. Skipping fit with {benchmark_model_name}.")
                    counts_fields_truth = None
                    return

            overdensities = np.array(overdensities)
            counts_fields_truth = np.array(counts_fields_truth)

            # Fit ngal vs delta relation.
            bias_bench_print(f"Fitting {benchmark_model.name} for res_{res_i}, mass_bin_{mass_bin_i}", verbose=True)
            fitted_params = benchmark_model.fit(overdensities, counts_fields_truth)

            if fitted_params is None:
                print(f"Was not able to fit model for res_{res_i}, mass_bin_{mass_bin_i}")
            else:
                # Poisson sample a realization using the mean value as the Poisson mu.
                predicted_count_fields = benchmark_model.predict(overdensities, fitted_params)
                n_benchmark_fits += 1

                for i, prediction in enumerate(predicted_count_fields):
                    counts_field_benchmark[i][res_i][mass_bin_i] = prediction

    bias_model_data.counts_field_benchmark = counts_field_benchmark

    print(f'Successfully completed {n_benchmark_fits} benchmark model fits.')
    # TODO: Add option to save predicted benchmark models to the corresponding hdf5 file
