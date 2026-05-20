import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
import natsort
from dataset_creation.createERA5Dataset import get_numpy_array, directional_encodings, get_max_min_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', type=str, required=True,
                        help='Path for each ERA5 input .grib files in chronological order.', nargs='+')
    parser.add_argument('-f', '--features', type=str, required=True,
                        help='Features from the ERA5 dataset to select. Note that features are alphabetically sorted in the numpy files.',
                        nargs='+')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path for output file.')
    parser.add_argument('-d', '--directional_encoding', type=str, default='degrees', choices=directional_encodings)
    parser.add_argument('-q', '--quotient', type=int, default=1,
                        help='Number to divide the latitude and longitude indices by.')
    args = parser.parse_args()

    file_path = Path(args.output_file)
    if not re.match(r'.*\([0-9]+\)\.npy$', args.output_file):
        dataset_files = [args.output_file]
    else:
        base_pattern = r'\([0-9]+\)\.npy$'
        base_name = re.sub(base_pattern, '', file_path.name)
        base_name = re.escape(base_name)  # need to escape because base_name might have special characters
        dataset_files = [f'{file_path.parent}{os.sep}{file}' for file in os.listdir(file_path.parent) if
                 re.match(rf'{base_name}{base_pattern}', file)]
    dataset_files = natsort.natsorted(dataset_files)

    csv_file = None
    for file in os.listdir(file_path.parent):
        if file.endswith('.csv'):
            csv_file = os.path.join(file_path.parent, file)
    if csv_file is None:
        raise FileNotFoundError()
    old_df = pd.read_csv(csv_file)
    new_df = get_max_min_df(args)
    df = pd.concat([new_df, old_df], ignore_index=True)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df.to_csv(csv_file)

    carry_over_rows_shape = (0, len(args.features), *(np.load(dataset_files[0]).shape[2:]))
    carry_over_rows = np.zeros(carry_over_rows_shape, dtype=np.float32)
    for np_file in dataset_files:
        np_dataset = np.load(np_file)
        total_rows = carry_over_rows
        if len(total_rows) > len(np_dataset):  # i.e. too many carry_over_rows
            carry_over_rows = total_rows[len(np_dataset):]
            total_rows = total_rows[:len(np_dataset)]
        print('start', len(total_rows))
        while len(total_rows) < len(np_dataset):
            num_rows_to_match = len(np_dataset) - len(total_rows)
            current_rows = get_numpy_array(args.input_files.pop(0), args)
            current_rows = current_rows[:, :, ::args.quotient, ::args.quotient]
            print(f'match {num_rows_to_match} with {len(current_rows)}')
            # case 1: too many rows
            if len(current_rows) > num_rows_to_match:
                print(f'too many: matching {num_rows_to_match} using {len(total_rows)} + {len(current_rows[:num_rows_to_match])} = {len(total_rows) + len(current_rows[:num_rows_to_match])}')
                print(total_rows.shape, current_rows.shape)
                total_rows = np.concatenate([total_rows, current_rows[:num_rows_to_match]], dtype=np.float32)
                carry_over_rows = current_rows[num_rows_to_match:]
            # case 2: exactly enough rows OR case 3: too few rows
            else:
                print(f'not enough/perfectly enough: concatenating {len(total_rows)} + {len(current_rows[:num_rows_to_match])} = {len(total_rows) + len(current_rows[:num_rows_to_match])}')
                print(total_rows.shape, current_rows.shape)
                total_rows = np.concatenate([total_rows, current_rows], dtype=np.float32)
                carry_over_rows = np.zeros(carry_over_rows_shape)
        print(f'finish: matching {len(total_rows)} with {len(np_dataset)}')
        np_dataset = np.concatenate([total_rows, np_dataset], axis=1, dtype=np.float32)
        np.save(np_file, np_dataset)


if __name__ == '__main__':
    main()
