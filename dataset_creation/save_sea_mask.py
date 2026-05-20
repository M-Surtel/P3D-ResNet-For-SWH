import argparse
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
import natsort

def get_dataset_files(file_name: str):
    if not re.match(r'.*\([0-9]+\)\.npy$', file_name):
        files = [file_name]
    else:
        file_path = Path(file_name)
        base_pattern = r'\([0-9]+\)\.npy$'
        base_name = re.sub(base_pattern, '', file_path.name)
        base_name = re.escape(base_name)  # need to escape because base_name might have special characters
        files = [f'{file_path.parent}{os.sep}{file}' for file in os.listdir(Path(file_name).parent) if re.match(rf'{base_name}{base_pattern}', file)]
    files = natsort.natsorted(files)
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--input-normalisation-file', type=str, required=True)
    parser.add_argument('-d', '--dataset-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True)
    args = parser.parse_args()

    dataset_files = get_dataset_files(args.dataset_file)

    df = pd.read_csv(args.input_normalisation_file)
    swh_index = df.index[df['feature'] == 'swh'][0]
    data_list = []
    for file in dataset_files:
        data = np.load(file)
        swh_data = data[:, swh_index]
        data_list.append(swh_data)
    cat_data = np.concat(data_list, axis=0)
    sea_mask = ~np.all(cat_data == 0, axis=0)
    np.save(args.output_file, sea_mask)


if __name__ == '__main__':
    main()
