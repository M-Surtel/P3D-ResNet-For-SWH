import numpy as np
from training.train import get_dataset_files
import argparse
import shutil
from pathlib import Path
import os
import torch


def parse_slice(slice_str: str):
    lat_str, lon_str = slice_str.split(',')
    lat = [eval(x) if x else None for x in lat_str.split(':')]
    lon = [eval(x) if x else None for x in lon_str.split(':')]
    return [slice(None), slice(None), slice(*lat), slice(*lon)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', type=str, required=True,
                                help="(Required) Path relative to this script to the dataset file."
                                     "\n If the filename conforms to the regular expression '*(0).npy', "
                                     "this script will automatically try to find files of '[Name](N).npy' "
                                     "until it reaches an N for which there is no file.")
    parser.add_argument("-n", "--input-normalisation-file", type=str,
                             help="If included min-max input normalisation will also be moved to output folder.")
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path for output file.')
    parser.add_argument('-s', '--slice', type=parse_slice, required=True,
                        help='Slice used to crop the spatial dimensions, expects input like "[lat_start]:[lat_stop], [lon_start]:[lon_stop]" (e.g. ":-5, 10:20")')
    args = parser.parse_args()

    output_folder = Path(args.output_file).parent
    os.makedirs(output_folder, exist_ok=True)

    # move input normalisation file
    if args.input_normalisation_file:
        norm_file = Path(args.input_normalisation_file)
        shutil.copy2(norm_file, output_folder / norm_file.name)

    input_files = get_dataset_files(args.input_files)
    # expects input shape of (T, F, Lat, Lon)
    for i, file in enumerate(input_files):
        # need data to be torch tensor because numpy slicing doesn't allow our notation
        data = torch.from_numpy(np.load(file))
        np.save(f'{args.output_file}({i})', data[args.slice])


if __name__ == '__main__':
    main()
