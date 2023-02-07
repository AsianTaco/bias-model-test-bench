import matplotlib.pyplot as plt
import numpy as np


def _compute_mean_and_variance(overdensity_field, count_field, ax, label):
    # Bins of overdensity.
    bins = 10 ** np.arange(-3, 3, 0.1)

    mean, var, bin_c = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = np.where((overdensity_field >= lo) & (overdensity_field < hi))
        mean.append(np.mean(count_field[mask]))
        var.append(np.var(count_field[mask]))
        bin_c.append((lo + hi) / 2.)

    ax.plot(bin_c, mean, label=f'mean ({label})')
    ax.plot(bin_c, var, label=f'var ({label})')


def plot_one_point_stats(bias_model_list, params):
    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        fig, axs = plt.subplots(2, figsize=(3, 5))

        overdensity_field_flat = bias_model_data.overdensity_field.flatten()

        n_gal_zero = 1

        try:
            count_field = bias_model_data.count_field
            axs[0].scatter(overdensity_field_flat, count_field.flatten() + n_gal_zero, label='predicted')
            _compute_mean_and_variance(overdensity_field_flat, count_field.flatten(), axs[1], 'predicted')
        except AttributeError:
            print("No predicted count field found in BiasModelData. Skipping plots")

        try:
            ground_truth = bias_model_data.count_field_truth
            axs[0].scatter(overdensity_field_flat, ground_truth.flatten() + n_gal_zero, label='ground truth', s=3)
        except AttributeError:
            print("No ground truth count field found in BiasModelData. Skipping plots")

        try:
            benchmark = bias_model_data.count_field_benchmark
            axs[0].scatter(overdensity_field_flat, benchmark.flatten() + n_gal_zero, label=benchmark_model_name, s=3)
            _compute_mean_and_variance(overdensity_field_flat, benchmark.flatten(), axs[1], benchmark_model_name)
        except AttributeError:
            print("No benchmark count field found in BiasModelData. Skipping plots")

        # Finalize figure.
        axs[0].axhline(n_gal_zero, label='$n_{gal}=0$')
        axs[0].set_xlabel(r"1 + $\delta$")
        axs[0].set_ylabel("Counts")
        axs[0].loglog()
        axs[0].set_xlim(1e-3, 1e3)
        axs[0].set_ylim(bottom=0)
        axs[0].legend()
        axs[1].set_xscale('log')
        axs[1].set_xlabel(r"1 + $\delta$")
        axs[1].legend()
        fig.suptitle(bias_model_name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"{params['out_dir']}/plots/one_point_{bias_model_name}.png")
