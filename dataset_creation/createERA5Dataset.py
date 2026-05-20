import argparse
import math
import numpy as np
import xarray as xr
import pandas as pd
import os
from pathlib import Path


def correct_naming(dataset):
    # some variables incorrectly have IDs instead of names, this function fixes that
    rename_dict = {"p140121": "swh1", "p140124": "swh2", "p140127": "swh3", "p140122": "mwd1", "p140125": "mwd2",
                   "p140128": "mwd3", "p140123": "mwp1", "p140126": "mwp2", "p140129": "mwp3", "p140209": "rhoao", "p140208": "wstar"}
    for variable in list(dataset.keys()):
        if variable in rename_dict.keys():
            dataset = dataset.rename({variable: rename_dict[variable]})
    return dataset


# for input normalisation
def get_max_min_df(args):
    # deep copy of args.features (don't want to change it)
    features = [value for value in args.features]
    # apply directional encoding
    if args.directional_encoding == 'sin_cos':
        direction_variables = ['mwd', 'mwd1', 'mwd2', 'mwd3', 'mdww', 'dwi', 'mdts']
        for dir_var in list(set(direction_variables) & set(features)):
            features.remove(dir_var)
            features.extend([f'{dir_var}_sin', f'{dir_var}_cos'])

    # put wave height as last index
    wave_heights = {'swh', 'swh1', 'swh2', 'swh3', 'shww', 'shts'}
    for wave_height in sorted(list(wave_heights & set(features))):
        features.pop(features.index(wave_height))
        features.append(wave_height)

    df = pd.DataFrame(columns=['feature', 'max', 'min'])
    for feature in features:
        feature_max, feature_min = feature_to_max_min[feature]
        new_row_df = pd.DataFrame([{'feature': feature, 'max': feature_max, 'min': feature_min}])
        df = pd.concat([df, new_row_df], ignore_index=True)
    return df


def replace_dir(dataset, args):
    direction_variables = {'mwd', 'mwd1', 'mwd2', 'mwd3', 'mdww', 'dwi', 'mdts'}
    features = [feature for feature in args.features]
    for dir_var in list(direction_variables & set(list(dataset.keys()))):
        direction_rad = np.deg2rad(dataset[dir_var] % 360)
        dataset[f'{dir_var}_sin'] = xr.DataArray(np.sin(direction_rad), dims=dataset[dir_var].dims, coords=dataset[dir_var].coords)
        dataset[f'{dir_var}_cos'] = xr.DataArray(np.cos(direction_rad), dims=dataset[dir_var].dims, coords=dataset[dir_var].coords)
        dataset = dataset.drop_vars([dir_var])
        features.remove(dir_var)
        features.extend([f'{dir_var}_sin', f'{dir_var}_cos'])
    return dataset, features


def get_numpy_array(dataset_file, args):
    dataset_list = []
    for feature in args.features:
        dataset_list.append(xr.load_dataset(dataset_file, engine='cfgrib', filter_by_keys={'shortName': feature}))
        if feature in ['u10', 'v10', 'u10n', 'v10n']:
            # reduce u10 and v10 to the standard spatial resolution
            dataset_list[-1] = dataset_list[-1].isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    dataset = xr.merge(dataset_list)
    any(ds.close() for ds in dataset_list)
    dataset = correct_naming(dataset)

    if args.directional_encoding == 'sin_cos':
        dataset, features = replace_dir(dataset, args)
    else:
        features = args.features

    # put wave height as last index
    wave_heights = {'swh', 'swh1', 'swh2', 'swh3', 'shww', 'shts'}
    for wave_height in sorted(list(wave_heights & set(features))):
        features.pop(features.index(wave_height))
        features.append(wave_height)

    np_array = np.stack([dataset[variable] for variable in features])
    dataset.close()
    np_array = np.nan_to_num(np_array)
    return np_array.transpose([1, 0, 2, 3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', type=str, required=True,
                        help='Path for each ERA5 input .grib files in chronological order!', nargs='+')
    parser.add_argument('-f', '--features', type=str, required=True,
                        help='Features from the ERA5 dataset to select. Note that features are alphabetically sorted in the numpy files.',
                        nargs='+')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path for output file.')
    parser.add_argument("-c", "--chunk-size", type=float, default=10,
                        help="The maximum number of gigabytes a NumPy array is allowed to be before it gets saved as a chunk (default = 10)")
    parser.add_argument('-d', '--directional_encoding', type=str, default='degrees', choices=directional_encodings)
    args = parser.parse_args()

    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    args.features = sorted(args.features)
    df = get_max_min_df(args)
    df.to_csv(f'{args.output_file}-feature-max-min.csv')

    np_dataset = None
    file_counter = 0
    for i, dataset_file in enumerate(args.input_files):
        print(f'{i}/{len(args.input_files)}')
        np_array = get_numpy_array(dataset_file, args)  # shape is (T, F, Lat, Lon)
        if np_dataset is None:
            np_dataset = np_array
        else:
            np_dataset = np.concatenate([np_dataset, np_array], axis=0)
        if np_dataset.size * 4 - args.chunk_size * 1e9 > 0 or dataset_file == args.input_files[-1]:
            # maximum chunk size reached or last file has been read
            np.save(f'{args.output_file}({file_counter})', np_dataset)
            file_counter += 1
            np_dataset = None


directional_encodings = ['degrees', 'sin_cos']
feature_to_max_min = \
    {
        'mwd':      [360.0,                 0.0],
        'mwd_sin':  [1.0,                   -1.0],
        'mwd_cos':  [1.0,                   -1.0],
        'mwp':      [1 / 0.03453,           0.0],  # T_max = 1 / f_min, T_min = 1 / f_max (but T_min lower than empirical min)
        'swh':      [30.0,                  0.0],  # empirically, waves won't be higher than 30m
        'rhoao':    [1.5,                   0.0],  # air density varies geographically, but for Chinese sea empirically won't exceed 1.5 kg / m**3
        'wstar':    [3.5,                   0.0],  # free convective flow is tied to air density (varies geographically) but empirically won't exceed 3.5
        'hmax':     [32.55601119995117,     0.0320071205496788],
        'mdts':     [360.0,                 0.0],
        'mdts_sin': [1.0,                   -1.0],
        'mdts_cos': [1.0,                   -1.0],
        'mdww':     [360.0,                 0.0],
        'mdww_sin': [1.0,                   -1.0],
        'mdww_cos': [1.0,                   -1.0],
        'mpts':     [1 / 0.03453,           0.0],  # T_max = 1 / f_min, T_min = 1 / f_max (but T_min lower than empirical min)
        'mpww':     [1 / 0.03453,           0.0],  # T_max = 1 / f_min, T_min = 1 / f_max (but T_min lower than empirical min)
        'msqs':     [0.1,                   0.0],  # empirically, mean square of slopes won't exceed 0.1
        'p2ps':     [1 / 0.03453,           0.0],
        'p2ww':     [1 / 0.03453,           0.0],
        'mp2':      [1 / 0.03453,           0.0],
        'wmb':      [999.0,                 0.0],  # max value we found
        'dwi':      [360.0,                 0.0],
        'dwi_sin':  [1.0,                   -1.0],
        'dwi_cos':  [1.0,                   -1.0],
        'wind':     [50,                    0.0],  # empirically, 10m neutral wind won't exceed  50m/s
        'pp1d':     [1 / 0.03453,           0.0],  # T_max = 1 / f_min, T_min = 1 / f_max (but T_min lower than empirical min)
        'tmax':     [1 / 0.03453,           0.0],  # T_max = 1 / f_min, T_min = 1 / f_max (but T_min lower than empirical min)
        'shts':     [30.0,                  0.0],  # empirically, waves won't be higher than 30m
        'shww':     [30.0,                  0.0],  # empirically, waves won't be higher than 30m
        'wdw':      [math.sqrt(2) + 0.01,   0.140703022480011],  # max should be sqrt(2) but numerics ig
        'dwps':     [math.sqrt(2) + 0.01,   0.0],  # max should be sqrt(2) but numerics ig
        'dwww':     [math.sqrt(2) + 0.01,   0.0],  # max should be sqrt(2) but numerics ig
        'bfi':      [10 + 0.01,            -10.0], # by definition must be between 10 and -10
        'wsk':      [1.0,                   -0.3301],  # -0.33 < C4 < 1.0 but numerics ig
        'wsp':      [44.00088119506836,     0.0],
        'wss':      [0.25,                  0.0],  # 0.0 < C3 < 0.25
        'u10n':     [50.0,                  -50.0],  # empirically, wind won't blow harder than 50m/s
        'v10n':     [50.0,                  -50.0],  # empirically, wind won't blow harder than 50m/s
        'sst':      [325.0,                 273.15],  # sst is in Kelvin and doesn't go below freezing, max is estimated empirically
    }

if __name__ == "__main__":
    main()
