import h5py


class BiasModelData:

    def __init__(self, fname):
        self.fname = fname

        self._load_bias_model_data()
        self.model_name = 'supermodel'

    def _load_bias_model_data(self):
        with h5py.File(self.fname, "r") as f:
            self.delta = f['delta'][...]
            self.galaxy_counts = f['galaxy_counts'][...]
            self.subhalo_counts = f['subhalo_counts'][...]
            # TODO: Read in meta data from h5 file

        print(self.delta.shape)
        print(self.galaxy_counts.shape)
        print(self.subhalo_counts.shape)


if __name__ == '__main__':
    BM = BiasModelData("../mock_data/eagle_25_box.hdf5")
