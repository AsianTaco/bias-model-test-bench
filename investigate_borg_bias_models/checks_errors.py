print("Loading libraries...")
from parameters import *
from forward_model import *

print("Done. Initializing BiasModelBORG for data_seed=0...")
BiasModel = BiasModelBORG("bias::BrokenPowerLaw")
i = 3

print("Done. Computing Poisson error for res=256...")
parameters_list = np.array([[2.5, 0.65, 1.5, 0.4]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list)
print(err)
parameters_list = np.array([[1, 1, 1, 1]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list)
print(err)
parameters_list = np.array([[2, .5, .1, .1]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list)
print(err)

print("Done. Computing KL error for res=256...")
parameters_list = np.array([[2.5, 0.65, 1.5, 0.4]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list, loss="KL")
print(err)
parameters_list = np.array([[1, 1, 1, 1]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list, loss="KL")
print(err)
parameters_list = np.array([[2, .5, .1, .1]]*6)
err = BiasModel.compute_fixed_res_error(i, parameters_list, loss="KL")
print(err)
print("Done.")