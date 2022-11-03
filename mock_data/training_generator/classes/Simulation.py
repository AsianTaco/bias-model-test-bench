import h5py
import numpy as np
from mpi4py import MPI
import argparse
from training_generator.cic import cic


class Simulation:
    def __init__(self):

        # MPI stuff.
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.rank
        self.comm_size = self.comm.size

        self._parse_command_line()

    def _parse_command_line(self):
        parser = argparse.ArgumentParser(description='Input options for generating mock bias data')
        parser.add_argument('--Ngrid', default=32, type=int)
        parser.add_argument('--eagle_db_user', default=None, type=str)
        parser.add_argument('--eagle_db_pass', default=None, type=str)
        parser.add_argument('--save_path', default="data.hdf5", type=str)
        parser.add_argument('--hdf5_name', default="arr", type=str)
        parser.add_argument('--which_field', default="counts", type=str)
        parser.add_argument('--which_count_field', default="galaxies", type=str)
        parser.add_argument('--which_count_type', default="both", type=str)

        args = parser.parse_args()
        for arg, arg_v in vars(args).items():
            setattr(self, arg, arg_v)

        assert self.which_field in ["counts", "density"]
        assert self.which_count_field in ['galaxies', 'subhaloes']
        assert self.which_count_type in ['centrals', 'satellites', 'both']

    def _grid_data(self, coords, name):

        print(f"[Rank {self.comm_rank}] Binning {name} particles...")
        if name == "density":
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

        # Reduce over cores.
        if self.comm_size > 1:
            if self.comm_rank == 0:
                H_recv = np.zeros_like(H)
            else:
                H_recv = None

            if name == "density":
                self.comm.Reduce([H, MPI.FLOAT], [H_recv, MPI.FLOAT], MPI.SUM, root=0)
            else:
                self.comm.Reduce([H, MPI.INT], [H_recv, MPI.INT], MPI.SUM, root=0)
            H = H_recv

        return H

    def _grid_density_data(self):

        # Load DM particles.
        coords = self.load_dark_matter_particles()

        # Grid.
        H = self._grid_data(coords, "density")

        # Convert to density contrast.
        if self.comm_rank == 0:
            H = np.true_divide(H, np.mean(H)) - 1

        return H

    def _grid_galaxy_data(self):

        # Load galaxies.
        coords = self.load_galaxies()
        
        # Grid.
        H = self._grid_data(coords, "galaxy")

        return H

    def run(self):

        if self.which_field == "density":
            # Compute density grid.
            grid = self._grid_density_data()
        else:
            grid = self._grid_galaxy_data()

        if self.comm_rank == 0:
            with h5py.File(self.save_path, "a") as f:
                g_name = f"{self.sim_name}/{self.hdf5_name}"

                if g_name in f:
                    del f[g_name]

                f.create_dataset(g_name, data=grid)

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
