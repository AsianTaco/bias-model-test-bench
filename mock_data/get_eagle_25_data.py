from pyread_eagle import EagleSnapshot
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load dark matter particles.
e25_dmo = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/PE/DMONLY/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
eagle = EagleSnapshot(e25_dmo)
bs = eagle.boxsize
eagle.select_region(0, bs, 0, bs, 0, bs)
coords = eagle.read_dataset(1, "Coordinates")

# Bin
Ngrid = 32
H, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Put into overdensity units.
H_mean = np.mean(H)
H /= H_mean
H -= 1

# Load galaxies.
import eagleSqlTools as sql

con = sql.connect("dmw381", password="ps320ysn")
myQuery = """select centreofpotential_x*0.6777 as x,
                centreofpotential_y*0.6777 as y,
                centreofpotential_z*0.6777 as z,
                masstype_star from eagle..refl0025n0376_subhalo where snapnum=28
                and masstype_star > 0"""
myData = sql.execute_query(con, myQuery)
coords = np.c_[myData["x"], myData["y"], myData["z"]]

# Bin
H_gal, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Load subhaloes.
myQuery = """select centreofpotential_x*0.6777 as x,
                centreofpotential_y*0.6777 as y,
                centreofpotential_z*0.6777 as z,
                masstype_star from eagle..refl0025n0376_subhalo where snapnum=28
                and masstype_dm > 0"""
myData = sql.execute_query(con, myQuery)
coords = np.c_[myData["x"], myData["y"], myData["z"]]

# Bin
H_sub, bin_edges = np.histogramdd(coords, bins=Ngrid)

# Plot
f, axarr = plt.subplots(1, 3)
axarr[0].imshow(np.sum(np.log10(1 + H), axis=2))
axarr[0].title.set_text("Density field")
axarr[1].imshow(np.sum(H_sub, axis=2))
axarr[1].title.set_text("Subhalo counts")
axarr[2].imshow(np.sum(H_gal, axis=2))
axarr[2].title.set_text("Galaxy counts")
# plt.show()

# Save to hdf5 file.
with h5py.File("eagle_25_box.hdf5", "w") as f:
    f.create_dataset("delta", data=H)
    f.create_dataset("galaxy_counts", data=H_gal)
    f.create_dataset("subhalo_counts", data=H_sub)

    # Header information.
    g = f.create_group("Header")
    g.attrs.create("BoxSize", bs)
    g.attrs.create("GridSize", Ngrid)
    g.attrs.create("GalaxyMassCut", 0.0)
    g.attrs.create("SubHaloMassCut", 0.0)
