from parameters import *
from forward_model import *
import time

# Evaluate time to compute the error for the full training set, with the different bias models:
reskeys=["res_0"]
fields_to_load=['dm_overdensity', 'counts_bin_0', 'counts_bin_1']
nkeys = len(reskeys)
nfields = len(fields_to_load)-1
parameters_list_dict = {"bias::Linear": np.array([[[1, 1]]*nfields]*nkeys),
                        "bias::BrokenPowerLaw": np.array([[[2.5, 0.65, 1.5, 0.4]]*nfields]*nkeys),
                        "bias::SecondOrderBias": np.array([[[1, 1, 1, 1]]*nfields]*nkeys)}

for bias_model in ["bias::Linear", "bias::BrokenPowerLaw", "bias::SecondOrderBias"]:
    print("####################\nComputing error for bias model: "+bias_model)
    start = time.time()
    BiasModel = BiasModelBORG(bias_model, reskeys=reskeys, fields_to_load=fields_to_load)
    parameters_list = parameters_list_dict[bias_model]
    nseeds = len(av_seeds)
    err = BiasModel.compute_error(parameters_list)
    end = time.time()
    print("Error: "+str(err))
    print("Time: "+str(end-start))
