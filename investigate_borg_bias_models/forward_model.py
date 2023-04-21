from parameters import *
import h5py
import numpy as np
import os
os.environ["PYBORG_QUIET"] = "yes"
import borg

def read_data(filename, seed, reskeys=None, fields=None):
    f = h5py.File(filename, mode='r')
    group = f[seed]
    data = []
    if reskeys is None:
        reskeys = group.keys()
    for res in reskeys:
        current_dict = {}
        for key in group[res].attrs.keys():
            current_dict[key] = group[res].attrs[key]
        if fields is None:
            fields = group[res].keys()
        for key in fields:
            current_dict[key] = {}
            current_dict[key]["values"] = np.array(group[res][key],dtype="float64")
            for key2 in group[res][key].attrs.keys():
                current_dict[key][key2] = group[res][key].attrs[key2]
        data.append(current_dict)
    f.close()
    return data

class BiasModelBORG:
    def __init__(self, model_name, filename=train, av_seeds=av_seeds, bins_keys=bins_keys,
                 reskeys=None, fields_to_load=None, default_data_seed=0):
        self.borg_model = model_name
        self.filename = filename
        self.av_seeds = av_seeds
        self.bins_keys = bins_keys
        self.reskeys = reskeys
        self.fields_to_load = fields_to_load
        self.current_data_seed = None # index of the current data seed, to avoid reloading data when not necessary
        self.load_data(default_data_seed)

    def load_data(self, data_seed=0):
        if self.current_data_seed != data_seed:
            self.current_data_seed = data_seed
            self.data = read_data(self.filename, self.av_seeds[data_seed], reskeys=self.reskeys, fields=self.fields_to_load)
            self.ndata = len(self.data) # number of resolutions
            self.NN = np.zeros(self.ndata, dtype=int)
            self.LL = np.zeros(self.ndata, dtype=float)
            self.bias_models = []
            for i in range(self.ndata):
                N = self.data[i]["ngrid"] ; L = self.data[i]["boxsize"]
                self.NN[i] = N ; self.LL[i] = L
                box = borg.forward.BoxModel()
                box.L = [L]*3 ; box.N = [N]*3
                bias_model = borg.forward.models.newModel(self.borg_model, box, {})
                bias_model.setName('bias_{}'.format(i))
                self.bias_models.append(bias_model)
        return self.ndata

    def get_N(self, i):
        return self.NN[i]

    def get_expected_count(self, i, parameters):
        """Compute the expected count field (Poisson intensity)"""
        bias_model = self.bias_models[i]
        bias_model.setModelParams({'biasParameters': parameters})
        bias_model.forwardModel_v2(self.data[i]["dm_overdensity"]["values"])
        count = np.empty(bias_model.getOutputBoxModel().N)
        bias_model.getDensityFinal(count)
        return count
    
    def get_count_field(self, i, parameters):
        count = self.get_expected_count(i, parameters)
        return np.random.poisson(count)
    
    def get_adjoint_gradient(self, i, dlogL_dcount):
        bias_model = self.bias_models[i]
        bias_model.adjointModel_v2(dlogL_dcount)
        ic = np.empty(bias_model.getBoxModel().N)
        bias_model.getAdjointModel(ic)
        return ic
        #TODO: check this
    
    def compute_fixed_res_error(self, i, parameters_list, loss="Poisson"):
        N = self.NN[i]
        err = 0
        Nbins = len(self.bins_keys)
        for bin, parameters in enumerate(parameters_list):
            count = self.get_expected_count(i, parameters)
            gt = self.data[i][self.bins_keys[bin]]["values"]
            if loss == "Poisson":
                count[count<=0] = 1e-10 ; gt[gt<=0] = 1e-10
                err += np.sum(-gt*np.log(count)+count)/Nbins
            elif loss == "mse":
                err += np.sqrt(np.mean((count-gt)**2))/Nbins
            elif loss == "KL":
                count[count<=0] = 1e-10 ; gt[gt<=0] = 1e-10
                err -= np.sum(gt*np.log(gt/count))/Nbins
            else:
                raise ValueError("Unknown error function")
        return err

    def compute_error(self, parameters_list, loss="Poisson"):
        err = 0
        nseeds = len(self.av_seeds)
        for data_seed in range(nseeds):
            self.load_data(data_seed)
            for i in range(self.ndata):
                err += self.compute_fixed_res_error(i, parameters_list[i], loss)/self.ndata
        return err/nseeds

    def plot_count_field(self, i, parameters, reskey, title=None):
        from seaborn import heatmap
        import matplotlib.pyplot as plt
        count = self.get_count_field(i, parameters)
        gt = self.data[i][reskey]["values"]
        N = self.NN[i]
        plt.figure(figsize=(10,5))
        if title is not None:
            plt.suptitle(title)
        plt.subplot(121)
        plt.title("Infered count field")
        heatmap(count[:,:,N//2], cmap="viridis")
        plt.subplot(122)
        plt.title("Ground truth")
        heatmap(gt[:,:,N//2], cmap="viridis")
        plt.show()
