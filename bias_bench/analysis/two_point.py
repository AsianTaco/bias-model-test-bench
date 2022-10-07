import matplotlib.pyplot as plt
import numpy as np


def plot_power_spectrum(BiasModelData, Lbox, kmin=1e-3, kmax=1.0, Nk=32, logk=False, normalize=True, even=True):

    delta = BiasModelData.delta
    gal_counts = BiasModelData.galaxy_counts

    k_density, power_density = compute_power_spectrum(delta, **meta)
    k_counts, power_counts = compute_power_spectrum(gal_counts, **meta)

    plt.loglog(k_density, power_density)
    plt.loglog(k_counts, power_counts)


def compute_power_spectrum(delta, Lbox, kmin=1e-3, kmax=1.0, Nk=32, logk=False, normalize=True, even=True):
    """Compute a 3d power spectrum from density contrast

    Args:
        delta (np.array): if complex, it is expected to be the fourier representation directly, otherwise
             a FFT will be run first
        Lbox (tuple of float or float):
        kmin (float, optional): Defaults to 1e-3 h/Mpc
        kmax (float, optional): Defaults to 1.0 h/Mpc
        Nk (int, optional): Defaults to 32.
        logk (bool, optional): Defaults to False
              *WARNING* logk=True needs checking
        normalize (bool, optional): Defaults to True. Apply proper normalization of the spectrum.
        even (bool, optional): If delta is complex, one needs to know if the real representation has odd or even last dimension.
    """
    na = np.newaxis

    N1, N2, N3 = delta.shape
    if delta.dtype == np.complex128:
        if even:
            N3 = (N3-1)*2
        else:
            N3 = (N3-1)*2+1
    N = N1, N2, N3

    Ntot = N1 * N2 * N3
    if type(Lbox) is float:
        Lbox = Lbox, Lbox, Lbox

    V = Lbox[0] * Lbox[1] * Lbox[2]

    ik = list(
        [np.fft.fftfreq(n, d=l / n) * 2 * np.pi for n, l in zip(N, Lbox)])

    ik[-1] = ik[-1][:(N[-1] // 2 + 1)]

    k_n = np.sqrt(ik[0][:, na, na]**2 + ik[1][na, :, na]**2 +
                  ik[2][na, na, :]**2)

    if delta.dtype == np.complex128:
      delta_hat = delta
    else:
      delta_hat = np.fft.rfftn(delta) * (V / Ntot)

    if logk:
        i_k_n = np.log10(k_n)
        i_k_min = np.log10(kmin)
        i_k_max = np.log10(kmax)
    else:
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
    if logk:
        b = 10**b
        bc = 10**bc
        H = H / bc
        raise ValueError("logk is not working")

    return bc, H
