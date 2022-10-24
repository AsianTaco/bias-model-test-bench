from pyread_eagle import EagleSnapshot
import numpy as np
import matplotlib.pyplot as plt
import h5py
import eagleSqlTools as sql
import sys

"""
Generate a density field and galaxy count field from the EAGLE simulations.

The density field input is from the *DARK MATTER ONLY* version of the
simulation and the galaxy count field is taken from the hydro version. This
then emulates what we do in BORG, i.e., going from DMO realisation to galaxy
count field.

Example usage:
    python3 generate_mock_eagle_data.py \
            <dmo_snapshot> \
            <eagle_db_username> \
            <eagle_db_password> \
            <Ngrid> \
            <savename>

    Ngrid is the grid dimension (resolution you want), i.e., 32, 64 etc...
    savename is the string tag attached to the files to identify them

On cosma the data is located in the dir:
    /cosma7/data/Eagle/ScienceRuns/Planck1/

e.g, the 25 box

/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/PE/DMONLY/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5

Output:
    This will make and HDF5 will 3 arrays (NxNxN). 
        - dm_delta_field (the overdensity field)
        - galaxy_count_field (the galaxy counts field)
        - subhalo_count_field (the subhalo counts field)

You could modify this script if you wished to add a mass cut to the
halo/galaxxy selections.
"""

snap_file = sys.argv[1]
database_username = sys.argv[2]
database_password = sys.argv[3]
Ngrid = int(sys.argv[4])
savename = sys.argv[5]

# Load dark matter particles.
print("Loading DM particles...")
eagle = EagleSnapshot(snap_file)
bs = eagle.boxsize
eagle.select_region(0, bs, 0, bs, 0, bs)
coords = eagle.read_dataset(1, "Coordinates")
print(f"Loaded {len(coords)} DM particles.")

# Bin
print("Binning DM particles")
H, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Put into overdensity units.
H_mean = np.mean(H)
H /= H_mean
H -= 1

# Load galaxies from eagle database.
print("Getting galaxies from database")
con = sql.connect(database_username, password=database_password)
myQuery = """select centreofpotential_x*0.6777 as x,
                centreofpotential_y*0.6777 as y,
                centreofpotential_z*0.6777 as z,
                masstype_star from eagle..refl0025n0376_subhalo where snapnum=28
                and masstype_star > 0"""
myData = sql.execute_query(con, myQuery)
coords = np.c_[myData["x"], myData["y"], myData["z"]]

# Bin
print("Binning galaxies from database")
H_gal, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Load subhaloes from eagle database.
print("Getting subhaloes from database")
myQuery = """select centreofpotential_x*0.6777 as x,
                centreofpotential_y*0.6777 as y,
                centreofpotential_z*0.6777 as z,
                masstype_star from eagle..refl0025n0376_subhalo where snapnum=28
                and masstype_dm > 0"""
myData = sql.execute_query(con, myQuery)
coords = np.c_[myData["x"], myData["y"], myData["z"]]

# Bin
print("Binning subhaloes from database")
H_sub, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Plot
print("Plotting")
f, axarr = plt.subplots(1, 3, figsize=(10,4))
axarr[0].imshow(np.sum(np.log10(1 + H), axis=2))
axarr[0].title.set_text("Density field")
axarr[1].imshow(np.sum(H_sub, axis=2))
axarr[1].title.set_text("Subhalo counts")
axarr[2].imshow(np.sum(H_gal, axis=2))
axarr[2].title.set_text("Galaxy counts")
plt.tight_layout(pad=0.1)
plt.savefig(f'{savename}.png')
plt.close()

print("Saving")
# Save to hdf5 file.
with h5py.File(f"{savename}.hdf5", "w") as f:
    f.create_dataset("dm_delta_field", data=H.astype(np.float32))
    f.create_dataset("subhalo_count_field", data=H_sub.astype(np.uint16))
    f.create_dataset("galaxy_count_field", data=H_gal.astype(np.uint16))

    # Header information.
    g = f.create_group("Header")
    g.attrs.create("BoxSize", bs)
    g.attrs.create("GridSize", Ngrid)
