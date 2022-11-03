import h5py
import numpy as np
from mpi4py import MPI

from read_swift import read_swift
from sibelius.galform import read_galform

from training_generator.classes.Simulation import Simulation

comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size

SAVE_DIR = "/cosma6/data/dp004/rttw52/GalaxyBiasTraining/"


class SibeliusDark(Simulation):
    def __init__(self):

        # Parameters for the Sibelius-DARK simulation.
        self.boxsize = 1000
        self.h = 1.0
        self.centre_point = np.array([499.34264252, 504.50740204, 497.31107545])
        self.load_region = 200
        self.sim_name = "SibeliusDARK"

    def mask_coords_to_loading_region(self, coords):
        # Mask to the loading region.
        mask = np.where(
            (coords[:, 0] > 0)
            & (coords[:, 0] <= self.load_region)
            & (coords[:, 1] > 0)
            & (coords[:, 1] <= self.load_region)
            & (coords[:, 2] > 0)
            & (coords[:, 2] <= self.load_region)
        )
        return mask

    def load_dark_matter_particles(self):

        fname = (
            "/cosma6/data/dp004/rttw52/SibeliusOutput/"
            "Sibelius_200Mpc_1/snapshots/Sibelius_200Mpc_1_0199/"
            "Sibelius_200Mpc_1_0199.0.hdf5"
        )

        swift = read_swift(fname, comm=comm)

        # Select region.
        region = self.get_loading_region()
        swift.select_region(1, *region)
        swift.split_selection()

        # Load particles.
        coords = swift.read_dataset(1, "Coordinates")
        print(f"[Rank {comm_rank}] Loaded {len(coords)} particles...")

        # Shift.
        coords -= self.centre_point
        coords += self.load_region / 2.0

        # Mask to the loading region.
        mask = self.mask_coords_to_loading_region(coords)
        coords = coords[mask]
        print(f"[Rank {comm_rank}] Clipped to {len(coords)} particles.")

        return coords

    def load_galaxies(self):

        nfiles = 1024
        output_no = 1
        gal_dir = (
            "/cosma7/data/dp004/jch/SibeliusOutput/Sibelius_200Mpc_1/"
            "Galform/models/Lacey16/output/"
        )
        mass_cut = 0

        galform = read_galform(gal_dir, nfiles, output_no, comm=comm)

        # Load galaxies.
        galform.load_galaxies(["xgal", "ygal", "zgal", "mstars_bulge", "mstars_disk"])
        galform.gather_galaxies()

        coords = np.c_[galform.data["xgal"], galform.data["ygal"], galform.data["zgal"]]
        print(f"[Rank {comm_rank}] Loaded {len(coords)} galaxies...")

        # Shift.
        coords -= self.centre_point
        coords += self.load_region / 2.0

        # Mask to the loading region.
        mask = self.mask_coords_to_loading_region(coords)
        coords = coords[mask]

        # Mask to mass cut
        mass = galform.data["mstars_disk"][mask] + galform.data["mstars_bulge"][mask]
        mask = np.where(mass >= mass_cut)
        coords = coords[mask]

        print(f"[Rank {comm_rank}] Clipped to {len(coords)} galaxies.")

        return coords


# Load the data.
data = SibeliusDark()
data.run()
