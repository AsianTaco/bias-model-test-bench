from bias_bench.io import BiasModelData
from bias_bench.analysis.two_point import plot_power_spectrum
from bias_bench.analysis.one_point import plot_one_point_stats
from bias_bench.analysis.three_point import plot_bispectrum


if __name__ == '__main__':
    BM = BiasModelData("mock_data/eagle_25_box_haloes.hdf5")

    plot_one_point_stats(BM)
    plot_power_spectrum(BM, kmin=1e-1, kmax=10.0, Nk=32, normalize=True,
                        show_density=False)
    plot_power_spectrum(BM, pylians=True)
    plot_bispectrum(BM)