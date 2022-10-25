import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpi4py import MPI
import sys
from training_generator.cic import cic


class Simulation:
    def __init__(self, do_cic=True):

        # MPI stuff.
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.rank
        self.comm_size = self.comm.size

        # CIC the density field?
        self.do_cic = do_cic

        # Gridsize for binning.
        self.Ngrid = int(sys.argv[1])

        # EAGLE database username/password.
        self.eagle_db_user = sys.argv[2]
        self.eagle_db_pass = sys.argv[3]

        # Load and grid galaxies in MPI?
        self.mpi_galaxy = True

    def _grid_data(self, coords, name):

        if coords is None:
            return None

        print(f"[Rank {self.comm_rank}] Binning {name} particles...")
        if name == "density" and self.do_cic:
            H = cic(coords, self.boxsize, self.Ngrid)
            H = np.ascontiguousarray(H).astype(np.float32)
        else:
            H, _ = np.histogramdd(
                coords,
                bins=[self.Ngrid, self.Ngrid, self.Ngrid],
                range=[
                    [0, self.load_region],
                    [0, self.load_region],
                    [0, self.load_region],
                ],
            )
            H = np.ascontiguousarray(H).astype(np.int32)

        if name == "galaxy" and self.galaxy_mpi == False:
            return H

        # Reduce over cores.
        if self.comm_size > 1:
            if self.comm_rank == 0:
                H_recv = np.zeros_like(H)
            else:
                H_recv = None

            if name == "density" and self.do_cic:
                self.comm.Reduce([H, MPI.FLOAT], [H_recv, MPI.FLOAT], MPI.SUM, root=0)
            else:
                self.comm.Reduce([H, MPI.INT], [H_recv, MPI.INT], MPI.SUM, root=0)
            H = H_recv

        return H

    def grid_density_data(self):

        # Load DM particles.
        coords = self.load_dark_matter_particles()

        # Grid.
        H = self._grid_data(coords, "density")

        # Convert to density contrast.
        if self.comm_rank == 0:
            H = np.true_divide(H, np.mean(H)) - 1

        return H

    def grid_galaxy_data(self):

        # Load galaxies.
        coords = self.load_galaxies()
        
        # Grid.
        H = self._grid_data(coords, "galaxy")

        return H

    def run(self):

        density_grid = self.grid_density_data()
        galaxy_grid = self.grid_galaxy_data()

        if self.comm_rank == 0:
            print(density_grid.shape, galaxy_grid.shape)
            assert density_grid.shape == galaxy_grid.shape
            
            self.plot(density_grid, galaxy_grid)

            print("Saving")
            with h5py.File(f"{self.sim_name}.hdf5", "w") as f:
                f.create_dataset("delta_dm_grid", data=density_grid)
                f.create_dataset("galaxy_count_grid", data=galaxy_grid)

    def plot(self, density_grid, galaxy_grid):

        f, axarr = plt.subplots(1, 2)
        mask = np.where(density_grid > 0)
        density_grid[mask] = np.log10(1+density_grid[mask])
        axarr[0].imshow(np.sum(density_grid, axis=2))
        axarr[1].imshow(np.sum(galaxy_grid, axis=2))
        plt.tight_layout(pad=0.1)
        plt.savefig(f"{self.sim_name}.png")
        plt.close()

    def get_loading_region(self):
        region = (
            self.centre_point[0] - self.load_region / 2.0,
            self.centre_point[0] + self.load_region / 2.0,
            self.centre_point[1] - self.load_region / 2.0,
            self.centre_point[1] + self.load_region / 2.0,
            self.centre_point[2] - self.load_region / 2.0,
            self.centre_point[2] + self.load_region / 2.0,
        )

        return region
