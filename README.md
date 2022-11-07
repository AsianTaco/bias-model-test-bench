Test bench
==

TODO
--
* More comprehensive documentation
* 

Installation
--

Build-in bias models
--

Input format
--

Expected HDF5 file which contains:

Expected data sets:

- Dark matter overdensity field $\rho / \bar{rho} - 1$
- Optional: Predicted count field ($N_{gal}$ or $N_{halo}$)
- Optional: Ground truth count field ($N_{gal}$ or $N_{halo}$)

Expected attributes:

- Dark matter overdensity dataset needs to provide box size ('BoxSize') and grid size ('GridSize') from the underlying
  simulation

Note:

* All fields are expected to have the same dimensions (N x N x N)

Plotting options
--

* 1pt statistics
* power spectrum
* bispectrum

Example usage:
--

* Standalone usage
* Module usage