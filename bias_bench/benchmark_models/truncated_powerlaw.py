import numpy as np
import scipy
from numba import njit
import matplotlib.pyplot as plt

from bias_bench.io import BiasModelData
from bias_bench.analysis.two_point import compute_power_spectrum


@njit
def _poisson_loop(delta_m, ngal_mean):
    assert len(delta_m.shape) == 1
    s = delta_m.shape[0]

    ngal = np.zeros_like(delta_m, dtype=np.int32)

    for i in range(s):
        ngal[i] = np.random.poisson(ngal_mean[i])

    return ngal


class PowerLawBiasModel:

    def power_law_model(self, delta_m, nmean, beta, rho_g, epsilon_g):
        d = 1 + delta_m
        x = rho_g * d ** (-epsilon_g)
        return nmean * d ** beta * np.exp(-x)

    def fit(self, delta, count_field):
        # Fit using SciPy curvefit.
        popt, pcov = scipy.optimize.curve_fit(self.power_law_model, delta, count_field)
        print(f"Power law bias fit params: {popt}")
        return popt

    def predict(self, delta, popt):
        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = self.power_law_model(delta, *popt)
        return ngal

    def sample(self, delta, popt):
        # Poisson sample ngal from delta_dm around ngal mean (and model params popt).
        return _poisson_loop(delta, self.predict(delta, popt))

        # mu for each delta.
        mu = self.predict(X, popt).values
        print(np.min(mu))
        # Return array.
        ngal = np.zeros_like(X['delta_dm'].values, dtype=np.int32)

        # Sample ngal for each delta, given a mu and fit parameters.
        alpha = 0.15
        var = mu + alpha * (mu ** 2) + 1e-5
        p = mu / var
        n = mu ** 2 / (var - mu)
        print(np.min(p), np.min(n))
        for i in scipy.tqdm(range(len(ngal))):
            try:
                ngal[i] = scipy.nbinom.rvs(n[i], p[i], size=1)
            except:
                continue
        return ngal


if __name__ == '__main__':
    BM = BiasModelData("../../mock_data/eagle_25_box.hdf5")

    pl_model = PowerLawBiasModel()
    delta_flattened = BM.delta.flatten()
    galaxy_counts_flattened = BM.galaxy_counts.flatten()

    fitted_params = pl_model.fit(delta_flattened, galaxy_counts_flattened)
    count_mean = pl_model.predict(delta_flattened, fitted_params)
    predicted_count_field = pl_model.sample(delta_flattened, fitted_params)

    # plt.scatter(delta_flattened, count_mean + 1, label='fitted')
    # plt.scatter(delta_flattened, galaxy_counts_flattened + 1, label='ground truth')
    # plt.scatter(delta_flattened, predicted_count_field + 1, label='predict')
    # plt.loglog()
    # plt.xlim(1e-3, 1e3)
    # plt.ylim(1, 1e2)
    # plt.legend()
    # plt.show()

    k_truth, power_truth = compute_power_spectrum(BM.galaxy_counts, 25., kmin=1e-1, kmax=10.0)
    k_predict, power_predict = compute_power_spectrum(predicted_count_field.reshape(32, 32, 32), 25., kmin=1e-1,
                                                      kmax=10.0)

    plt.loglog(k_truth, power_truth, label="truth")
    plt.loglog(k_predict, power_predict, label='predict')
    # TODO: Add ground truth plot
    plt.legend()
    plt.show()
