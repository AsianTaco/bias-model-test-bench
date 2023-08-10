import numpy as np
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


class LinearBias(BenchmarkModel):

    def __init__(self):
        super().__init__("Linear Bias", 2)
        self.bounds = [(1e-6, 1e8), (1e-6, 1e8)]

    def get_mean_ngal(self, rho, nmean, b_1):
        """
        Using the truncated power law bias model, compute the mean number of
        galaxies expected for a given density.

        Incoming density field is expected to be 1D, i.e., the flattened 3D array.

        Parameters
        ----------
        rho : ndarray
            The 3D density field
        nmean : float
            ...
        b_1 : float
            ...

        Returns
        -------
        ngal_mean : ndarray
            Expected number of galaxies for the given densities
        """

        nmean_bounds = self.bounds[0]
        if nmean < nmean_bounds[0] or nmean > nmean_bounds[1]:
            return np.NaN

        b_0_bounds = self.bounds[1]
        if b_1 < b_0_bounds[0] or b_1 > b_0_bounds[1]:
            return np.NaN

        # Return array
        ngal_mean = np.zeros_like(rho)

        # Non zero densites
        idx = np.where(rho > -1.0)

        d = 1 + rho[idx]
        ngal_mean[idx] = nmean * (1 + (b_1 * d))

        return ngal_mean

    def predict(self, delta, popt):
        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = self.get_mean_ngal(delta.flatten(), *popt)

        if np.any(np.isnan(ngal)):
            return np.NaN

        return ngal.reshape(delta.shape)
