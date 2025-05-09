# Project plan to move forward with testbench

## Vision of testbench

* Collabative tool to check model performace for halo/galaxy bias
* Gives comparable metrics for different models (unified plotting style)
* A collection of essential metrics to test bias model on
* Curated data sets for testing that are well thought through for different test scenarios

## Deliverable products

* LtU internal list of bias projects that can be plugged. Depending on the specific use-case
* Paper unifying different bias projects to make them comparable for the community

## Metrics

  [] Include marginal halo mass function check in the 1 point if possible
  [] Add ensemble simulation plot for higher order statistics check e.g. Shivam's plots or BAM paper plots

## Features

  [] Html visualization capabilities
  [] Add field visualization (overdensity vs. predicted count field vs. true count field)
  [] Add halo catalogue input capabilities (Reading in the pure catalog and do binning into mass bins etc. on the fly)
  [] Fix emcee sampling for bias models with explicit likelihoods
  [] Add more entry points for helper functions (Formatting halo catalogs)
  [] Export more utility functions that can then be used by the testbench as a package

## Collaborative work

  [] Include tarp as a model check if desired (Talk to Pablo)
  [] Add curated galaxy count data set for testing use standard HOD as benchmark (Talk to Lucia)
  [] Add dataset for galaxy counts for quijote dataset
  [] Demonstration in LtU internal calls
  [] Host potential workshop for paper project with testbench
