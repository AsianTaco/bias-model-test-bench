import pandas as pd
import numpy as np


def convert_dataframe_catalog_to_numpy(dataframe_cat):
    numpy_halo_cat = np.empty(dataframe_cat.index.size,
                              dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('mvir', '<f8')])

    numpy_halo_cat['x'] = dataframe_cat.values.T[0]
    numpy_halo_cat['y'] = dataframe_cat.values.T[1]
    numpy_halo_cat['z'] = dataframe_cat.values.T[2]
    numpy_halo_cat['mvir'] = dataframe_cat.values.T[3]

    return numpy_halo_cat


def extract_halo_cat_with_mass_threshold(halo_cat, mass_threshold, capital_id=False, convert_to_numpy=False):
    if capital_id:
        reduced_cat = halo_cat.loc[halo_cat['Mvir'] >= mass_threshold, ['X', 'Y', 'Z', 'Mvir']]
    else:
        reduced_cat = halo_cat.loc[halo_cat['mvir'] >= mass_threshold, ['x', 'y', 'z', 'mvir']]

    if convert_to_numpy:
        reduced_cat = convert_dataframe_catalog_to_numpy(reduced_cat)

    return reduced_cat


def read_rockstar_ascii_cat(cat_paths, mass_threshold, capital_id=False, convert_to_numpy=False):
    reduced_cats = []

    for cat_path in cat_paths:
        halo_cat = pd.read_csv(cat_path, sep=' ', comment='#',
                               names=['ID', 'DescID', 'Mvir', 'Vmax', 'Vrms', 'Rvir', 'Rs', 'Np', 'X', 'Y',
                                      'Z', 'VX', 'VY', 'VZ', 'JX', 'JY', 'JZ', 'Spin', 'rs_klypin',
                                      'Mvir_all', 'M200b', 'M200c', 'M500c', 'M2500c', 'Xoff', 'Voff',
                                      'spin_bullock', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]', 'A[z]',
                                      'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', 'A[y](500c)',
                                      'A[z](500c)', 'T/|U|', 'M_pe_Behroozi', 'M_pe_Diemer',
                                      'Halfmass_Radius', 'PID'])
        reduced_cat = extract_halo_cat_with_mass_threshold(halo_cat, mass_threshold, capital_id, convert_to_numpy)
        reduced_cats.append(reduced_cat)

    return reduced_cats
