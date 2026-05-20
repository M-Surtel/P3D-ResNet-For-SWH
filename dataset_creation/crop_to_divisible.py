import torch
import numpy as np
import argparse
from pathlib import Path
import shutil
from training.train import get_dataset_files
import os


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
    parser.add_argument('-d', '--divisible-by', type=int, required=True,
                        help='Forces both spatial outputs to be divisible by this number.')
    parser.add_argument('--lat-start', action='store_true',
                        help="Takes indices off the latitude's start, instead off the latitude's end.")
    parser.add_argument('--lon-start', action='store_true',
                        help="Takes indices off the longitude's start, instead off the longitude's end.")
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
        data = torch.from_numpy(np.load(file))

        # calculate indices
        indices = [slice(None), slice(None)]
        _, _, lats, lons = data.shape
        if args.lat_start:  # i.e. data[n:]
            indices += [slice(lats - int(lats / args.divisible_by) * args.divisible_by, None)]
        else:  # i.e. data[:n]
            indices += [slice(None, int(lats / args.divisible_by) * args.divisible_by)]
        if args.lon_start:  # i.e. data[n:]
            indices += [slice(lons - int(lons / args.divisible_by) * args.divisible_by, None)]
        else:  # i.e. data[:n]
            indices += [slice(None, int(lons / args.divisible_by) * args.divisible_by)]

        np.save(f'{args.output_file}({i})', data[indices])


if __name__ == '__main__':
    main()
