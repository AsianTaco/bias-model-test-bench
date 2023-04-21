print("Loading libraries...")
from parameters import *
from forward_model import *
from matplotlib import pyplot as plt
import seaborn as sns
print("Done. Initializing BiasModelBORG for data_seed=0...")
BiasModel = BiasModelBORG("bias::BrokenPowerLaw")
print("Done. Performing forward pass for res=256...")
i = 3
parameters = np.array([2.5, 0.65, 1.5, 0.4])
count = BiasModel.get_count_field(i, parameters)
N = BiasModelBORG.get_N(BiasModel, i)
print("Done. Plotting slice_ref...")
slice_ref = count[:,:,N//2]
sns.heatmap(slice_ref)
plt.savefig(wd+"results/slice_ref.png")
print("Done. Saving slice_ref...")
np.save(wd+"results/slice_ref.npy", slice_ref)
print("Done.")
