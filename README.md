Test bench
==

## Installation


### Conda

If you use conda, you can set up a new environment and install the dependencies by running:

```bash
conda env create -f environment.yml
```

Then install as an editable package via

```bash
pip install -e .
```

### Pip only

If you are using a pure pip setup, make sure to have the correct python version installed (See `requires-python` field in the **pyproject.toml** file).
Then again, install all dependencies and the package using

```bash
pip install -e .
```

### Test installation

You can test the installation by running the entry point script via

```bash
galaxy-bias-benchmark examples/eagle_example.yml
```

If the script ran successfully, you should be able to inspect some plots located at **examples/plots**.

Build-in bias models
--

The test bench already comes with some implemented bias models that can be used out of the box.

### Truncated power law (https://arxiv.org/abs/1309.6641)

A non-linear local bias model parameterized by 4 parameters by Neyrinck, M., et al. (2014)

Input format
--

Expected HDF5 file which contains:

Expected data sets:

- Dark matter over-density field $\rho / \bar{\rho} - 1$
- Optional: Predicted count field ($ N_{gal} \text{or} N_{halo} $)
- Optional: Ground truth count field ($ N_{gal} \text{or} N_{halo} $)

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
* **TBD**: Quantitative metrics (log-likelihood, KL) 

Example usage:
--

TBD