"""
Create galaxy bias training data from the eagle simulations.

Example usage
-------------

python3 run.py <Ngrid> <database_username> <database_password>

    - Ngrid is the grid dimension to bin the data (eg 32,64..).
    - database_username is EAGLE database username.
    - database_password is EAGLE database password.

Will create an HDF5 file with the DM overdensity field (CIC smoothed) and a
galaxy count field (no smoothing). 

Can be run over MPI.
"""

from training_generator.classes.Eagle import Eagle25, Eagle100


#x = Eagle25()
#x.run()

x = Eagle100()
x.run()
