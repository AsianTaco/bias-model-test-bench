import matplotlib.pyplot as plt
import numpy as np

from bias_bench.io import BiasModelData


def plot_density_field(bias_model_data: BiasModelData, params, benchmark_model_name):
    """
    Plot the projected dark matter overdensity field
    """

    f, axarr = plt.subplots(2, 2, figsize=(6,6))

    img = np.sum(bias_model_data.overdensity_field, axis=2)

    axarr[0,0].imshow(img)

    if hasattr(bias_model_data, "count_field_benchmark"):
        img = np.sum(bias_model_data.count_field_benchmark, axis=2)
        axarr[0,1].imshow(img)

    if hasattr(bias_model_data, "count_field"):
        img = np.sum(bias_model_data.count_field, axis=2)
        axarr[1,0].imshow(img)

    if hasattr(bias_model_data, "count_field_truth"):
        img = np.sum(bias_model_data.count_field_truth, axis=2)
        axarr[1,1].imshow(img)

    for i, lab in enumerate([r'$\delta$', '$N_{\mathrm{bench}}$', '$N_{\mathrm{pred}}$', '$N_{\mathrm{true}}$']):
        axarr[i//2,i%2].set_axis_off()

        axarr[i//2,i%2].text(0.1, 0.9, lab, transform=axarr[i//2,i%2].transAxes,
            fontsize=10, c='white')

    plt.tight_layout(pad=0.1)
    plt.show()
