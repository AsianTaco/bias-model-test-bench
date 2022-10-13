import matplotlib.pyplot as plt
import numpy as np


def plot_power_spectrum(BiasModelData, Lbox, kmin=1e-1, kmax=10.0, Nk=32, logk=False, normalize=True):

    delta = BiasModelData.delta
    gal_counts = BiasModelData.galaxy_counts_flattened
    meta = BiasModelData.model_name

    k_density, power_density = compute_power_spectrum(delta, Lbox, kmin=kmin, kmax=kmax, Nk=Nk, logk=logk, normalize=normalize)
    k_counts, power_counts = compute_power_spectrum(gal_counts, Lbox, kmin=kmin, kmax=kmax, Nk=Nk, logk=logk, normalize=normalize)

    plt.loglog(k_density, power_density, label="density")
    plt.loglog(k_counts, power_counts, label='counts')
    # TODO: Add ground truth plot
    plt.legend()
    plt.show()


# TODO: Find light-weight third-party power spectrum & bispectrum estimator
def compute_power_spectrum(delta, Lbox, kmin=1e-3, kmax=1.0, Nk=32, logk=False, normalize=True):
    """Compute a 3d power spectrum from density contrast

    Args:
        delta (np.array): if complex, it is expected to be the fourier representation directly, otherwise
             a FFT will be run first
        Lbox (float):
        kmin (float, optional): Defaults to 1e-3 h/Mpc
        kmax (float, optional): Defaults to 1.0 h/Mpc
        Nk (int, optional): Defaults to 32.
        logk (bool, optional): Defaults to False
              *WARNING* logk=True needs checking
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

    k_n = np.sqrt(ik[0][:, na, na]**2 + ik[1][na, :, na]**2 +
                  ik[2][na, na, :]**2)

    delta_hat = np.fft.rfftn(delta) * (V / Ntot)

    i_k_n = k_n
    i_k_min = kmin
    i_k_max = kmax

    # TODO: Handle nyquist correctly
    Hw, _ = np.histogram(i_k_n, range=(i_k_min, i_k_max), bins=Nk)
    H, b = np.histogram(i_k_n,
                        weights=delta_hat.real**2 + delta_hat.imag**2,
                        range=(i_k_min, i_k_max),
                        bins=Nk)

    H = H / Hw
    H[Hw==0] = 0
    if normalize:
      H /= V
    bc = 0.5 * (b[1:] + b[:-1])

    return bc, H
