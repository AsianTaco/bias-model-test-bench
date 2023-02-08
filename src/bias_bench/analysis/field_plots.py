from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from bias_bench.data_io import BiasModelData


def plot_density_field(bias_model_list: Sequence[BiasModelData], params):
    """
    Plot the projected dark matter overdensity field
    """

    for bias_model_index, bias_model_data in enumerate(bias_model_list):
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))

        bias_model_name = params[f'bias_model_{bias_model_index + 1}']['name']
        benchmark_model_name = params[f'bias_model_{bias_model_index + 1}']['count_field_benchmark_name']

        img = np.sum(bias_model_data.overdensity_field, axis=2)

        axs[0, 0].imshow(img)

        if hasattr(bias_model_data, "count_field_benchmark"):
            img = np.sum(bias_model_data.count_field_benchmark, axis=2)
            axs[0, 1].imshow(img)

        if hasattr(bias_model_data, "count_field"):
            img = np.sum(bias_model_data.count_field, axis=2)
            axs[1, 0].imshow(img)

        if hasattr(bias_model_data, "count_field_truth"):
            img = np.sum(bias_model_data.count_field_truth, axis=2)
            axs[1, 1].imshow(img)

        for i, lab in enumerate([r'$\delta$', '$N_{\mathrm{bench}}$', '$N_{\mathrm{pred}}$', '$N_{\mathrm{true}}$']):
            axs[i // 2, i % 2].set_axis_off()

            axs[i // 2, i % 2].text(0.1, 0.9, lab, transform=axs[i // 2, i % 2].transAxes,
                                      fontsize=10, c='white')

        fig.suptitle(bias_model_name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"{params['out_dir']}/plots/density_fields_{bias_model_name}.png")
