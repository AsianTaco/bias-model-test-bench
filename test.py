from bias_bench.io import BiasModelData
from bias_bench.analysis.two_point import plot_power_spectrum

BM = BiasModelData("../mock_data/eagle_25_box.hdf5")

plot_power_spectrum(BM)
