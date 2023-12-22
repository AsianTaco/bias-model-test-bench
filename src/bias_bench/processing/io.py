import pandas as pd
import numpy as np

from bias_bench.constants import capital_rockstar_columns


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


def read_rockstar_ascii_cat(cat_paths, mass_threshold, capital_id=False, convert_to_numpy=False, infer_header=True):
    reduced_cats = []

    for cat_path in cat_paths:
        # TODO: bypass this way of hard coding the header columns as constants
        header_names = capital_rockstar_columns
        header_line = 0 if infer_header else None
        # FIXME: passing heading names messes up headers with inferred column names
        halo_cat = pd.read_csv(cat_path, sep=' ', comment='#', names=header_names, header=header_line)
        reduced_cat = extract_halo_cat_with_mass_threshold(halo_cat, mass_threshold, capital_id, convert_to_numpy)
        reduced_cats.append(reduced_cat)

    return reduced_cats
