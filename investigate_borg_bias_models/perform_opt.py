from parameters import *
from forward_model import *
from os.path import exists
import time
from scipy.optimize import minimize

import argparse
parser = argparse.ArgumentParser(description='Perform optimization.')
parser.add_argument('--bias_model', type=str, default="BrokenPowerLaw",
                    help='Bias model to use. Available: Linear, BrokenPowerLaw, SecondOrderBias.')
parser.add_argument('--id', type=str, default="v0.0",
                    help='ID of the run. The otpimized parameters will be saved in wd+str(nkeys)+"_"+str(nfields)+"_"+bias_model+"_opt_parameters_"+id+.npy".')
parser.add_argument('--nkeys', type=int, default=1,
                    help='Number of resolutions to use (1 for 32^3 only, ..., 4 for all res up to 256^3).')
parser.add_argument('--nfields', type=int, default=1,
                    help='Number of mass bins of halo counts to fit.')
parser.add_argument('--force', type=bool, default=False,
                    help='If True, the optimization will be performed even if the results already exist.')

args = parser.parse_args()
bias_model = "bias::"+args.bias_model
id = args.id
nkeys = args.nkeys
nfields = args.nfields
force = args.force
reskeys = ["res_{}".format(i) for i in range(nkeys)]
fields_to_load = ['dm_overdensity'] + ["counts_bin_{}".format(i) for i in range(nfields)]

parameters_list_dict = {"bias::Linear": np.array([[[1]*2]*nfields]*nkeys),
                        "bias::BrokenPowerLaw": np.array([[[1]*4]*nfields]*nkeys),
                        "bias::SecondOrderBias": np.array([[[1]*4]*nfields]*nkeys)}

# TODO: define the correct physical bounds for the parameters:
bounds = {"bias::Linear": [(1., 500.), (0.01, 3.)]*nfields*nkeys,
         "bias::BrokenPowerLaw": [(1., 500.), (0.01, 3.), (.01, 2.), (.01, 2.)]*nfields*nkeys,
         "bias::SecondOrderBias": [(1., 500.), (0.01, 3.), (.01, 2.), (.01, 2.)]*nfields*nkeys}

print("Defining bias model...")
BiasModel = BiasModelBORG(bias_model, reskeys=reskeys, fields_to_load=fields_to_load)
parameters_list = parameters_list_dict[bias_model]

def objective(parameters_list_flat):
    parameters_list = parameters_list_flat.reshape((nkeys, nfields, -1))
    return BiasModel.compute_error(parameters_list, loss="Poisson")

def callback(parameters_list_flat): # only used to add verbosity at each iteration
    parameters_list = parameters_list_flat.reshape((nkeys, nfields, -1))
    l = BiasModel.compute_error(parameters_list, loss="Poisson")
    print("Parameters: {}".format(parameters_list))
    print("Loss: {}".format(l))

initial_guess = parameters_list.flatten()

if __name__ == '__main__':
    res_path = wd+"results/"+str(nkeys)+'_'+str(nfields)+'_'+bias_model+"_opt_parameters_"+id+".npy"
    if not exists(res_path) or force:
        print("Done. Starting optimization...")
        t0 = time.time()
        result = minimize(objective, initial_guess, method="L-BFGS-B", callback=callback,
                        tol=1e-6,
                        bounds=bounds[bias_model],
                        options={'disp': True, 'eps':1e-5, 'finite_diff_rel_step': 1e-5})
        print("Done. Time elapsed: {}. Saving results...".format(time.time()-t0))
        np.save(res_path, result.x.reshape((nkeys, nfields, -1)))

        print(result)
        print(result.x.reshape((nkeys, nfields, -1)))
        print("Done.")
    else:
        print("Results already exist. Skipping optimization.")
