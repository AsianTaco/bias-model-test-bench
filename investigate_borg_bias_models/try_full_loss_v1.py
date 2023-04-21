from parameters import *
from forward_model import *
import time

# Evaluate time to compute the error for the full training set, with the different bias models:
parameters_list_dict = {"bias::Linear": np.array([[[1, 1]]*6]*4),
                        "bias::BrokenPowerLaw": np.array([[[2.5, 0.65, 1.5, 0.4]]*6]*4),
                        "bias::SecondOrderBias": np.array([[[1, 1, 1, 1]]*6]*4)}

for bias_model in ["bias::Linear", "bias::BrokenPowerLaw", "bias::SecondOrderBias"]:
    print("####################\nComputing error for bias model: "+bias_model)
    start = time.time()
    BiasModel = BiasModelBORG(bias_model)
    parameters_list = parameters_list_dict[bias_model]
    err = BiasModel.compute_error(parameters_list)
    end = time.time()
    print("Error: "+str(err))
    print("Time: "+str(end-start))
