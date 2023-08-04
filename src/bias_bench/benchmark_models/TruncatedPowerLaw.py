import numpy as np
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


class TruncatedPowerLaw(BenchmarkModel):

    def __init__(self):
        super().__init__("Truncated Power Law", 4)
        self.bounds = [(1., 1e8), (1e-6, 6.), (1e-6, 3.), (1e-6, 1e5)]

    def get_mean_ngal(self, rho, nmean, beta, epsilon_g, rho_g):
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
        beta : float
            ...
        epsilon_g
            ...
        rho_g : float
            ...

        Returns
        -------
        ngal_mean : ndarray
            Expected number of galaxies for the given densities
        """

        # TODO: refactor this

        nmean_bounds = self.bounds[0]
        if nmean < nmean_bounds[0] or nmean > nmean_bounds[1]:
            return np.NaN

        beta_bounds = self.bounds[1]
        if beta < beta_bounds[0] or beta > beta_bounds[1]:
            return np.NaN

        epsilon_g_bounds = self.bounds[2]
        if epsilon_g < epsilon_g_bounds[0] or epsilon_g > epsilon_g_bounds[1]:
            return np.NaN

        rho_g_bounds = self.bounds[3]
        if rho_g < rho_g_bounds[0] or rho_g > rho_g_bounds[1]:
            return np.NaN

        # Return array
        ngal_mean = np.zeros_like(rho)

        # Non zero densites
        idx = np.where(rho > -1.0)

        d = 1 + rho[idx]
        x = (d / rho_g) ** (-epsilon_g)
        ngal_mean[idx] = nmean * d ** beta * np.exp(-x)

        return ngal_mean

    def predict(self, delta, popt):
        # TODO: implement bounds explicitly

        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = self.get_mean_ngal(delta.flatten(), *popt)

        if np.any(np.isnan(ngal)):
            return np.NaN

        return ngal.reshape(delta.shape)
