from bias_bench.io import BiasModelData
from bias_bench.Params import BiasParams
from bias_bench.analysis.one_point import plot_one_point_stats
from bias_bench.analysis.three_point import plot_bispectrum
from bias_bench.analysis.two_point import plot_power_spectrum

import matplotlib.pyplot as plt


def plot_bias_model_metrics(bias_model_data: BiasModelData, bias_params: BiasParams):
    params = bias_params.data

    plt.style.use(f"./plot_styles/{params['plotting_style']}")

    # Make plots.
    if 'ngal_vs_rho' in params['plots']:
        plot_one_point_stats(bias_model_data)
    if 'power_spectrum' in params['plots']:
        plot_power_spectrum(bias_model_data, params['power_spectrum'])
    if 'bi_spectrum' in params['plots']:
        plot_bispectrum(bias_model_data, params['bi_spectrum'])
