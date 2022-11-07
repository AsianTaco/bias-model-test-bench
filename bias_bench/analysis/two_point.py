import Pk_library as PKL

import numpy as np
import matplotlib.pyplot as plt

def compute_power_spectrum(field, l_box, MAS):
    # FIXME: Enable use of double precision
    overdensity = np.array(field, dtype=np.float32)

    pk = PKL.Pk(overdensity, l_box, axis=0, MAS=MAS)

    return pk.k3D, pk.Pk[:, 0]


def plot_power_spectrum(bias_model_list, params):

    fig, axs = plt.subplots(2, figsize=(3, 5))

    for ii, bias_model_data in enumerate(bias_model_list):
        ground_truth_field_exists = hasattr(bias_model_data, 'count_field_truth')
    
        #if ground_truth_field_exists:
        #    fig, axs = plt.subplots(2, figsize=(3, 5))
        #else:
        #    fig, axs = plt.subplots(1)
   
        l_box = bias_model_data.info['BoxSize']
        show_density = params['power_spectrum']['show_density']
        MAS = params['power_spectrum']['MAS']
    
        if show_density:
            overdensity_field = bias_model_data.overdensity_field
            k_density, power_density = compute_power_spectrum(overdensity_field, l_box, MAS=MAS)
            axs[0].loglog(k_density, power_density, label="density")
    
        try:
            ground_truth = bias_model_data.count_field_truth
            ground_truth = ground_truth / np.mean(ground_truth) - 1
            k_truth, power_truth = compute_power_spectrum(ground_truth, l_box, MAS=MAS)
            axs[0].loglog(k_truth, power_truth, label='ground truth')
        except AttributeError:
            print("No ground truth count field found in BiasModelData. Skipping plots")
    
        try:
            count_field = bias_model_data.count_field
            k_counts, power_counts = compute_power_spectrum(count_field, l_box, MAS=MAS)
            axs[0].loglog(k_counts, power_counts, label='predicted')
    
            if ground_truth_field_exists:
                axs[1].loglog(k_truth, power_counts / power_truth, label='predicted')
        except AttributeError:
            print("No predicted count field found in BiasModelData. Skipping plots")
    
        try:
            benchmark = bias_model_data.count_field_benchmark
            benchmark = benchmark / np.mean(benchmarck) - 1
            k_benchmark, power_benchmark = compute_power_spectrum(benchmark, l_box, MAS=MAS)
            axs[0].loglog(k_benchmark, power_benchmark, label=params[f'bias_model_{ii+1}']['name'])
    
            if ground_truth_field_exists:
                axs[1].loglog(k_truth, power_benchmark / power_truth, label='benchmark')
        except AttributeError:
            print("No benchmark count field found in BiasModelData. Skipping plots")

    axs[0].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    axs[0].set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    axs[1].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    axs[1].set_ylabel(r"$P(k) / P_{truth}(k)$ ")
    axs[1].axhline(1, linewidth=.5, linestyle='--', color='black')
    axs[1].set_ylim(1e-1, 1e1)
    fig.tight_layout(pad=0.1)
    plt.legend()
    plt.show()
