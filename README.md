# Test bench

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

If you are using a pure pip setup, make sure to have the correct python version installed (See `requires-python` field
in the **pyproject.toml** file).
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

## Usage

For a fully automatic run using the command line, the test bench expects a HDF5 file containing the data and a configuration YAML-file.
Checkout the **mock_data/** and **example/** folders for some examples.

### HDF5 file structure

The structure is expected to be as follows:

| #  Layer | Contains  | Description                                                                                                                                      | Has attributes                             | Naming Convention                                                                                        |
|:--------:|:---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|:---------------------------------------------------------------------------------------------------------|
|    1     |  Groups   | Divides the different simulations <br /> e.g. random seed or pipeline                                                                            | None                                       | Arbitrary prefix followed by integers <br /> e.g. 'quijote_HR_0', 'quijote_HR_1',...                     |
|    2     |  Groups   | Divides the different gridding resolutions                                                                                                       | **boxsize** <br /> **ngrid**               | **Enforced** to follow 'res_0', 'res_1',...                                                              |
|    3     | Data sets | Contains the underlying dark matter overdensity field and the corresponding halo count fields (ground truth or predicted) at different mass bins | **M_hi** and **M_lo** for the count fields | **Enforced** to follow: <br /> 'dm_overdensity' <br /> 'counts_bin_{i}' <br /> 'counts_predicted_bin_{i} |

#### About the attributes

* 'boxsize' refers to the length of the simulation box given in units of Mpc/h
* 'ngrid' refers to the number of bins per side
* 'M_hi' and 'M_lo' indicates the upper and lower bound of the mass bin as 10^{M_hi} or 10^{M_lo} in units of the solar mass

Note:

* All fields are expected to have the same dimensions (N x N x N)

### YAML config file

The config file specifies the setup of the HDF5 file that is to be digested and also controls the plotting setup.
A comprehensive config file with all possible fields can be found in at **example/full_config.yml**

## Build-in bias models

The test bench already comes with some implemented bias models that can be used out of the box.

### Truncated power law (https://arxiv.org/abs/1309.6641)

A non-linear local bias model parameterized by 4 parameters by Neyrinck, M., et al. (2014)

## Plotting options

* 1pt statistics
* power spectrum
* bispectrum
* **TBD**: Quantitative metrics (log-likelihood, KL)

Package usage:
--

We want to provide some more standalone functions of the package as well in the future.
The idea would be that the test bench can also be imported and used as a package, e.g.

```python
import bias_bench as bb

truncated_powerlaw_params = bb.fit_neyrinck(dm_overdensity, counts_bin_1)
```