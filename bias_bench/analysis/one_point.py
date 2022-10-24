import matplotlib.pyplot as plt

from bias_bench.io import BiasModelData


def plot_one_point_stats(bias_model_data: BiasModelData):

    overdensity_field_flat = bias_model_data.overdensity_field.flatten()

    try:
        count_field = bias_model_data.count_field
        plt.scatter(overdensity_field_flat, count_field.flatten() + 1, label='predicted')
    except AttributeError:
        print("No predicted count field found in BiasModelData. Skipping plots")

    try:
        ground_truth = bias_model_data.count_field_truth
        plt.scatter(overdensity_field_flat, ground_truth.flatten() + 1, label='ground truth', s=3)
    except AttributeError:
        print("No ground truth count field found in BiasModelData. Skipping plots")

    # Finalize figure.
    plt.xlabel(r"1 + $\delta$") 
    plt.ylabel("Counts")
    plt.loglog()
    plt.xlim(1e-3, 1e3)
    plt.ylim(1, 1e2)
    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.show()
