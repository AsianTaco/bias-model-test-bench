import numpy as np
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


class TruncatedPowerLaw(BenchmarkModel):

    def __init__(self):
        super().__init__("Truncated Power Law", 4)
        self.bounds = [(1e-6, 1e8), (1e-6, 6.), (1e-6, 3.), (1e-6, 1e5)]

    def get_mean_ngal(self, delta_m, nmean, beta, epsilon_g, rho_g):
        """
        Using the truncated power law bias model (Neyrinck et al. 2014) to computez
        the mean number of galaxies expected for given over-density :math:`\\delta_m`:

        .. math::
            n_{gal} = n_{mean} \\rho_m^\\beta \\exp[(\\rho_m / \\rho_g)^{-\\epsilon}],

        where :math:`\\rho_m = \\delta_m + 1`.

        Incoming density field is expected to be 1D, i.e., the flattened 3D array.

        Parameters
        ----------
        delta_m : ndarray
            The 3D density field
        nmean : float
            First bias parameter controlling the overall amplitude of the function.
        beta : float
            ...
        epsilon_g : float
            ...
        rho_g : float
            ...

        Returns
        -------
        ngal_mean : np.ndarray
            Expected number of galaxies for the given densities.
        """

        if self.check_bounds(np.asarray([nmean, beta, epsilon_g, rho_g])):
            return np.NaN

        # # result array
        # ngal_mean = np.zeros_like(delta_m)
        #
        # # Non zero densites
        # idx = np.where(delta_m > -1.0)
        #
        # d = 1 + delta_m[idx]
        # x = (d / rho_g) ** (-epsilon_g)
        # ngal_mean[idx] = nmean * d ** beta * np.exp(-x)

        rho_m = 1 + delta_m + 1e-6
        x = (rho_m / rho_g) ** (-epsilon_g)
        ngal_mean = nmean * (rho_m ** beta) * np.exp(-x) + 1e-6

        return ngal_mean

    def adj_gradient_mean_ngal(self, delta_m, gradient, nmean, alpha, epsilon, rho_g):
        rho_m = 1 + delta_m + 1e-6
        a = rho_m ** (alpha - 1)
        b = (rho_m / rho_g) ** (-epsilon)
        f = np.exp(-b)

        return (alpha + epsilon * b) * a * f * nmean * gradient

    def predict(self, delta, popt):
        # TODO: implement bounds explicitly

        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = self.get_mean_ngal(delta.flatten(), *popt)

        if np.any(np.isnan(ngal)):
            return np.NaN

        return ngal.reshape(delta.shape)
