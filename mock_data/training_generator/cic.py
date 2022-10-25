import numpy as np


def get_index(i, j, k, Ngrid):
    """Generate 1D index."""

    return (i % Ngrid) + ((j % Ngrid) * Ngrid) + ((k % Ngrid) * Ngrid * Ngrid)


def cic(coords, boxsize, Ngrid, ndim=3):

    # Create a new grid which will contain the densities
    grid = np.zeros(Ngrid**ndim, dtype=np.float32)

    # Bin coords into their cells and find distance to cell centre.
    x_c = np.floor(coords[:, 0] * Ngrid / boxsize).astype(np.int32)
    y_c = np.floor(coords[:, 1] * Ngrid / boxsize).astype(np.int32)
    z_c = np.floor(coords[:, 2] * Ngrid / boxsize).astype(np.int32)

    # Distance to center of cell
    d_x = coords[:, 0] * Ngrid / boxsize - (x_c + 0.5)
    d_y = coords[:, 1] * Ngrid / boxsize - (y_c + 0.5)
    d_z = coords[:, 2] * Ngrid / boxsize - (z_c + 0.5)

    # Which side of the center is this particle on
    inc_x = np.ones_like(x_c)
    inc_x[d_x < 0] = -1
    inc_y = np.ones_like(y_c)
    inc_y[d_y < 0] = -1
    inc_z = np.ones_like(z_c)
    inc_z[d_z < 0] = -1

    # Work with absolute values.
    d_x = np.abs(d_x)
    d_y = np.abs(d_y)
    d_z = np.abs(d_z)

    t_x = 1.0 - d_x
    t_y = 1.0 - d_y
    t_z = 1.0 - d_z

    # Add contributions to 8 cells
    idx = get_index(x_c, y_c, z_c, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=t_x * t_y * t_z)

    idx = get_index(x_c + inc_x, y_c, z_c, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=d_x * t_y * t_z)

    idx = get_index(x_c, y_c + inc_y, z_c, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=t_x * d_y * t_z)

    idx = get_index(x_c, y_c, z_c + inc_z, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=t_x * t_y * d_z)

    idx = get_index(x_c + inc_x, y_c + inc_y, z_c, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=d_x * d_y * t_z)

    idx = get_index(x_c, y_c + inc_y, z_c + inc_z, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=t_x * d_y * d_z)

    idx = get_index(x_c + inc_x, y_c, z_c + inc_z, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=d_x * t_y * d_z)

    idx = get_index(x_c + inc_x, y_c + inc_y, z_c + inc_z, Ngrid)
    grid[:] += np.bincount(idx, minlength=Ngrid**ndim, weights=d_x * d_y * d_z)

    return grid.reshape(Ngrid, Ngrid, Ngrid, order="F")
