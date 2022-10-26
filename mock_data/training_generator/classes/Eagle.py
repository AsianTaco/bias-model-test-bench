import numpy as np

from pyread_eagle import EagleSnapshot
import eagleSqlTools as sql

from training_generator.classes.Simulation import Simulation


class Eagle100(Simulation):
    def __init__(self):

        super().__init__()

        # Parameters for the EAGLE 100 simulation.
        self.galaxy_mpi = False
        self.boxsize = 100 * 0.6777
        self.h = 0.6777
        self.sim_name = "refl0100n1504"
        self.load_region = self.boxsize
        self.dmo_snap = (
            "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/"
            "PE/DMONLY/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
        )

    def load_dark_matter_particles(self):
        """Load all DM particles from DMO simulation snapshot z=0."""

        # Load dark matter particles.
        eagle = EagleSnapshot(self.dmo_snap)
        bs = eagle.boxsize
        eagle.select_region(0, bs, 0, bs, 0, bs)
        if self.comm_size > 1:
            eagle.split_selection(self.comm_rank, self.comm_size)
        coords = eagle.read_dataset(1, "Coordinates")
        print(f"Rank {self.comm_rank} Loaded {len(coords)} DM particles.")

        return coords

    def load_galaxies(self):
        """Load all galaxies from the EAGLE database."""

        if self.comm_rank != 0:
            return None

        # Load galaxies from eagle database.
        print("Getting galaxies from database")
        con = sql.connect(self.eagle_db_user, password=self.eagle_db_pass)
        myQuery = f"""
                select
                    centreofpotential_x*{self.h} as x,
                    centreofpotential_y*{self.h} as y,
                    centreofpotential_z*{self.h} as z
                from 
                    eagle..{self.sim_name}_subhalo
                where
                    snapnum=28
                    and masstype_star >= 1e8
                    and subgroupnumber = 0
                """
        myData = sql.execute_query(con, myQuery)
        coords = np.c_[myData["x"], myData["y"], myData["z"]]
        print(f"[Rank {self.comm_rank}] Loaded {len(coords)} galaxies...")

        return coords


class Eagle25(Eagle100):
    def __init__(self):

        super().__init__()

        # Parameters for the Sibelius-DARK simulation.
        self.boxsize = 25 * 0.6777
        self.h = 0.6777
        self.sim_name = "refl0025n0376"
        self.load_region = self.boxsize
        self.dmo_snap = (
            "/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/"
            "PE/DMONLY/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
        )

class Eagle50(Eagle100):
    def __init__(self):

        super().__init__()

        # Parameters for the Sibelius-DARK simulation.
        self.boxsize = 50 * 0.6777
        self.h = 0.6777
        self.sim_name = "refl0050n0752"
        self.load_region = self.boxsize
        self.dmo_snap = (
            "/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/"
            "PE/DMONLY/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
        )

