import matplotlib.pyplot as plt
import numpy as np

from bias_bench.io import BiasModelData


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


def plot_one_point_stats(bias_model_data: BiasModelData, benchmark_model_name):
    f, axarr = plt.subplots(2)

    overdensity_field_flat = bias_model_data.overdensity_field.flatten()

    try:
        count_field = bias_model_data.count_field
        axarr[0].scatter(overdensity_field_flat, count_field.flatten() + 1, label='predicted')
        _compute_mean_and_variance(overdensity_field_flat, count_field.flatten() + 1, axarr[1], 'predicted')
    except AttributeError:
        print("No predicted count field found in BiasModelData. Skipping plots")

    try:
        ground_truth = bias_model_data.count_field_truth
        axarr[0].scatter(overdensity_field_flat, ground_truth.flatten() + 1, label='ground truth', s=3)
    except AttributeError:
        print("No ground truth count field found in BiasModelData. Skipping plots")

    try:
        benchmark = bias_model_data.count_field_benchmark
        axarr[0].scatter(overdensity_field_flat, benchmark.flatten() + 1, label=benchmark_model_name, s=3)
        _compute_mean_and_variance(overdensity_field_flat, benchmark.flatten() + 1, axarr[1], benchmark_model_name)
    except AttributeError:
        print("No benchmark count field found in BiasModelData. Skipping plots")

    # Finalize figure.
    axarr[0].set_xlabel(r"1 + $\delta$")
    axarr[0].set_ylabel("Counts")
    axarr[0].loglog()
    axarr[0].set_xlim(1e-3, 1e3)
    axarr[0].set_ylim(1, 1e2)
    axarr[0].legend()
    axarr[1].set_xscale('log')
    axarr[1].set_xlabel(r"1 + $\delta$")
    axarr[1].legend()
    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.show()
