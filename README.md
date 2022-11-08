Test bench
==

TODO
--
* More comprehensive documentation
* 

Installation
--

If you use conda, you can simply set up a new environment and install the dependencies by running:

```bash
conda env create -f environment.yml
```

Initialize the package for the scripts using

```bash
pip install -e .
```

If you are using a pure python/pip setup, make sure to have the correct python version installed (See `python_requires` field in the **setup.cfg** file).
Then again, install all dependencies and the package using the above `pip` command.

You can test the installation by running the example script

```bash
python examples/run_standalone.py examples/eagle_25_example.yml
```

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