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

    counts_field_benchmark = []
    n_benchmark_fits = 0
    for sim_i in range(bias_model_data.n_simulations):
        counts_field_benchmark_per_res = []

        for res_i in range(bias_model_data.n_res):
            overdensity_flat = bias_model_data.dm_overdensity_fields[sim_i][res_i].flatten()
            counts_field_benchmark_per_res_and_mass_bin = []

            for mass_bin_i in range(bias_model_data.n_mass_bins):

                try:
                    counts_flat = bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i].flatten()
                except IndexError:
                    print(f"No truth count field found. Skipping fit with {benchmark_model_name}.")

                if benchmark_model_name == "truncated_power_law":
                    pl_model = TruncatedPowerLaw()

                    # Fit ngal vs delta relation.
                    bias_bench_print(f"Fitting model for sim_{sim_i}, res_{res_i}, mass_bin_{mass_bin_i}", verbose=True)
                    # TODO: Implement try-expect to make this faster
                    # TODO: batch optimise over the differently seeded simulations
                    fitted_params = pl_model.fit(overdensity_flat, counts_flat)
                    print(f"Fitted params: \n {fitted_params}")

                    if fitted_params is None:
                        print(f"Was not able to fit model for sim_{sim_i}, res_{res_i}, mass_bin_{mass_bin_i}")
                        predicted_count_field = None
                    else:
                        # Poisson sample a realization using the mean value as the Poisson mu.
                        predicted_count_field = pl_model.sample(overdensity_flat, fitted_params).reshape(
                            bias_model_data.count_fields_truth[sim_i][res_i][mass_bin_i].shape)
                        n_benchmark_fits += 1

                    # Store predicted count field.
                    counts_field_benchmark_per_res_and_mass_bin.append(predicted_count_field)
            counts_field_benchmark_per_res.append(counts_field_benchmark_per_res_and_mass_bin)
        counts_field_benchmark.append(counts_field_benchmark_per_res)

    bias_model_data.counts_field_benchmark = counts_field_benchmark

    print(f'Successfully completed {n_benchmark_fits} benchmark model fits.')
    # TODO: Add option to save predicted benchmark models to the corresponding hdf5 file
