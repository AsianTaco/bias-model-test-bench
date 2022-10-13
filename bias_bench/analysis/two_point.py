import matplotlib.pyplot as plt
import numpy as np

from bias_bench.io import BiasModelData


def plot_power_spectrum(bias_model_data: BiasModelData, kmin=1e-3, kmax=1.0, Nk=32, normalize=True,
                        show_density=False):
    l_box = bias_model_data.info['BoxSize']

    try:
        count_field = bias_model_data.count_field
        k_counts, power_counts = compute_power_spectrum(count_field, l_box, kmin, kmax, Nk, normalize)
        plt.loglog(k_counts, power_counts, label='predicted')
    except AttributeError:
        print("No predicted count field found in BiasModelData. Skipping plots")

    try:
        ground_truth = bias_model_data.count_field_truth
        k_truth, power_truth = compute_power_spectrum(ground_truth, l_box, kmin, kmax, Nk, normalize)
        plt.loglog(k_truth, power_truth, label='truth')
    except AttributeError:
        print("No ground truth count field found in BiasModelData. Skipping plots")

    if show_density:
        overdensity_field = bias_model_data.overdensity_field
        k_density, power_density = compute_power_spectrum(overdensity_field, l_box, kmin, kmax, Nk, normalize)
        plt.loglog(k_density, power_density, label="density")

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.suptitle("Power spectrum comparison of count fields")
    plt.title(f"Predicted count field generated by {bias_model_data.model_name}")
    plt.legend()
    plt.show()


# TODO: Find light-weight third-party power spectrum & bispectrum estimator
def compute_power_spectrum(delta, Lbox, kmin, kmax, Nk, normalize):
    """Compute a 3d power spectrum from density contrast

    Args:
        delta (np.array): 3D field for which the power-spectrum is to be computed
        Lbox (float): Box size of the field in Mpc/h
        kmin (float, optional): Defaults to 1e-3 h/Mpc
        kmax (float, optional): Defaults to 1.0 h/Mpc
        Nk (int, optional): Defaults to 32.
        normalize (bool, optional): Defaults to True. Apply proper normalization of the spectrum.
    """
    na = np.newaxis

    N1, N2, N3 = delta.shape
    N = N1, N2, N3

    Ntot = N1 * N2 * N3
    Lbox = Lbox, Lbox, Lbox

    V = Lbox[0] * Lbox[1] * Lbox[2]

    ik = list(
        [np.fft.fftfreq(n, d=l / n) * 2 * np.pi for n, l in zip(N, Lbox)])

    ik[-1] = ik[-1][:(N[-1] // 2 + 1)]

    k_n = np.sqrt(ik[0][:, na, na] ** 2 + ik[1][na, :, na] ** 2 +
                  ik[2][na, na, :] ** 2)

    delta_hat = np.fft.rfftn(delta) * (V / Ntot)

    i_k_n = k_n
    i_k_min = kmin
    i_k_max = kmax

    # TODO: Handle nyquist correctly
    Hw, _ = np.histogram(i_k_n, range=(i_k_min, i_k_max), bins=Nk)
    H, b = np.histogram(i_k_n,
                        weights=delta_hat.real ** 2 + delta_hat.imag ** 2,
                        range=(i_k_min, i_k_max),
                        bins=Nk)

    H = H / Hw
    H[Hw == 0] = 0
    if normalize:
        H /= V
    bc = 0.5 * (b[1:] + b[:-1])

    return bc, H
