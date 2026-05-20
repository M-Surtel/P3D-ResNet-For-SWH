import functools
import torch
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from typing import Dict
import re

from model_classes.abstract_model import ParentModel
from model_classes.persistence_model import PersistenceModel
from model_classes.P3D_ResNet import P3DResNet
from model_classes.unet import UNet
from training.train import parse_model_args
from training.train import get_dataset_files


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    if torch.cuda.is_available():
        print("CUDA is available!  Testing on GPU...")
        device = torch.device('cuda')
    else:
        print("CUDA not available, testing on CPU...")
        device = torch.device('cpu')
    torch.set_default_device(device)
    model_map: Dict[str, type[ParentModel]] = {
        "Persistence": PersistenceModel,
        "P3D-ResNet": P3DResNet,
        "U-Net": UNet,
    }

    parser.add_argument(
        "-m", "--model",
        metavar="MODEL",
        nargs=1,
        required=True,
        help="(Required) Choose which model to train.",
        choices=model_map.keys()
    )
    parser.add_argument(
        "-a", "--model-args",
        nargs='+',
        help=f"Provide space-separated arguments for the initialisation of the chosen model."
             f"\nOnly supply arguments for the chosen model class.",
    )
    parser.add_argument('-c', '--checkpoint', type=str, help='Model checkpoint to load state dictionary from.')
    parser.add_argument('--data', type=str, nargs=1, required=True,
                        help="(Required) Path relative to this script to the dataset file."
                             "\n If the filename conforms to the regular expression '*(0).npy', "
                             "this script will automatically try to find files of '[Name](N).npy' "
                             "until it reaches an N for which there is no file.")
    parser.add_argument("-b", "--batch-size", type=int, help="Size of each training batch (Default is 1)", default=1)
    parser.add_argument("-d", "--debugging-file", type=str,
                        help="File name relative to this script where the debugging information will be saved "
                             "(debugging prints will only be saved if this option is included).")
    parser.add_argument("-o", "--output", type=str,
                              help="Path relative to this script where the .pt error dictionary will be saved.")
    parser.add_argument("--pixels", type=str, nargs='+', required=True,
                        help="Indices of pixels in the output area that will be recorded."
                             "Must be in the form '([lat_idx],[lon_idx])' (e.g. '(12,34) (56,78)').")
    parser.add_argument("-n", "--input-normalisation-file", type=str,
                        help="(Required) min-max input normalisation is applied using the min-max value provided in file."
                             "Input normalisation file is needed for determining indices of input/output variables."
                             "Should probably implement way to run this script without max/min normalisation.")
    parser.add_argument('-i', '--input-features', type=str, nargs='+', required=True,
                        help="(Required) Names of chosen input variables as they appear in the user-supplied normalisation file.")
    parser.add_argument('-u', '--output-features', type=str, nargs='+', required=True,
                        help="(Required) Names of chosen output variables as they appear in the user-supplied normalisation file.")
    parser.add_argument('-v', '--error-output-features', type=str, nargs='+', required=True,
                        help="(Required) Names of chosen output variables that contribute to the error as they appear in the user-supplied normalisation file.")
    args = parser.parse_args()

    Path(args.debugging_file).parent.mkdir(exist_ok=True, parents=True)
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)

    model_class: type[ParentModel] = model_map[args.model[0]]
    with open(args.debugging_file, 'a') as file:
        print(f"\n\n\n[Chosen model is {model_class}]\n", file=file)

    # load model
    model_args = parse_model_args(args.model_args)

    # input normalisation
    df = pd.read_csv(args.input_normalisation_file)
    max_min_list = [(df[df['feature'] == feature]['max'].to_list()[0], df[df['feature'] == feature]['min'].to_list()[0])
                    for feature in args.input_features]

    target_feature_name = args.error_output_features[0]
    input_feature_idx = args.input_features.index(target_feature_name)

    with open(args.debugging_file, 'a') as file:
        print("[main]: instantiate model, loss, optimiser", file=file)
    if torch.cuda.is_available():
        model: ParentModel = model_class(df['feature'].to_list(), args.input_features, args.output_features,
                                         **model_args, max_min_list=max_min_list).cuda()
    else:
        model: ParentModel = model_class(df['feature'].to_list(), args.input_features, args.output_features,
                                         **model_args, max_min_list=max_min_list)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # load data
    test_data = get_dataset_files(args.data[0])

    # pixels of interest
    pattern = re.compile(r'\((\d+)\s*,\s*(\d+)\)')
    pixel_indices = []
    for pixel_str in args.pixels:
        matches = pattern.findall(pixel_str)
        for match in matches:
            lat_idx = int(match[0])
            lon_idx = int(match[1])
            pixel_indices.append((lat_idx, lon_idx))

    error_dict = {
            (lat_idx, lon_idx): {
                'predicted_diff': torch.zeros(0),
                'actual_diff': torch.zeros(0)
            }
            for (lat_idx, lon_idx) in pixel_indices
        }

    def batch(files, batch_size, preprocessing):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(error_dict):
                input_file_counter = 0

                file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device) if input_file_counter < len(
                    files) else None
                input_file_counter += 1

                starting_index = 0  # a file's 1st batch doesn't often start at the file's 1st index
                while file2 is not None:
                    file1 = file2
                    file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device) if input_file_counter < len(
                        files) else None
                    input_file_counter += 1
                    length = 0
                    spatial_length = 0

                    # file1 batches
                    for batch_start in range(starting_index, len(file1), batch_size):
                        batch_x: torch.Tensor
                        batch_y: torch.Tensor
                        batch_end = batch_start + batch_size + model.lookback + model.lead_time
                        # batches using 1 file
                        if batch_end <= len(file1):
                            batch_x, batch_y = preprocessing(file1[batch_start:batch_end])
                            error_dict = func(batch_x, batch_y)
                        # batches using 2 files
                        elif file2 is not None:
                            input_file1 = file1[batch_start:]
                            input_file2 = file2[:batch_end - len(file1)]
                            if len(input_file1) + len(
                                    input_file2) < batch_size + model.lookback + model.lead_time and input_file_counter < len(
                                    files):
                                file3 = torch.from_numpy(np.load(files[input_file_counter])).to(device)
                                input_file_counter += 1
                                input_file3 = file3[:batch_end - len(file1) - len(file2)]
                                input_data = torch.cat([input_file1, input_file2, input_file3])
                                batch_x, batch_y = preprocessing(input_data)
                                error_dict = func(batch_x, batch_y)
                                file2 = file3
                                break
                            else:
                                input_data = torch.cat([input_file1, input_file2])
                                batch_x, batch_y = preprocessing(input_data)
                                error_dict = func(batch_x, batch_y)
                                starting_index = batch_start + batch_size - len(file1)
                        elif batch_end - batch_size + model.lead_time <= len(
                                file1):  # if enough data left for >= 1 point, create batch
                            batch_x, batch_y = preprocessing(file1[batch_start:])
                            error_dict = func(batch_x, batch_y)
                    del file1
                return error_dict
            return wrapper
        return decorator

    @batch(files=test_data, batch_size=args.batch_size, preprocessing=model.preprocess_data)
    def calculate_errors(inputs: torch.Tensor, labels: torch.Tensor):
        outputs = model(inputs)

        assert len(outputs.shape) == 4
        outputs = outputs[:, error_output_indices]
        labels = labels[:, error_output_indices]
        # if there is a single error output feature, feature dimension might collapse
        if len(outputs.shape) < 4:
            outputs = torch.unsqueeze(outputs, 1)
        if len(labels.shape) < 4:
            labels = torch.unsqueeze(labels, 1)

        inputs = model.crop_output(inputs[:, input_feature_idx, -1])  # select most recent temporal index
        outputs = model.swh_reduce(outputs)
        labels = model.swh_reduce(labels)

        predicted_diff = outputs - inputs
        actual_diff = labels - inputs
        for (lat_idx, lon_idx) in pixel_indices:
            error_dict[lat_idx, lon_idx]['predicted_diff'] = (
                torch.cat(
                    [
                        error_dict[lat_idx, lon_idx]['predicted_diff'],
                        predicted_diff[:, lat_idx - 4, lon_idx - 4] # - 4 for border
                    ]
                )
            )
            error_dict[lat_idx, lon_idx]['actual_diff'] = (
                torch.cat(
                    [
                        error_dict[lat_idx, lon_idx]['actual_diff'],
                        actual_diff[:, lat_idx - 4, lon_idx - 4] # - 4 for border
                    ]
                )
            )
        return error_dict

    # determine output features we care about the error for
    error_output_indices = [args.output_features.index(error_output_feature) for error_output_feature in
                            args.error_output_features]

    target_feature_name = args.error_output_features[0]
    input_feature_idx = args.input_features.index(target_feature_name)
    errors = calculate_errors(error_dict)

    with open(args.debugging_file, 'a') as file:
        for (lat_idx, lon_idx) in pixel_indices:
            print(f"For latitude and longitude pixels of {lat_idx}, and {lon_idx}:\n"
                  f"\tPredicted difference averages {torch.mean(errors[lat_idx, lon_idx]['predicted_diff'])} "
                  f"with {len(errors[lat_idx, lon_idx]['predicted_diff'])} items.\n"
                  f"\tActual difference averages {torch.mean(errors[lat_idx, lon_idx]['actual_diff'])} "
                  f"with {len(errors[lat_idx, lon_idx]['actual_diff'])} items.\n", file=file)

    torch.save(errors, args.output)


if __name__ == '__main__':
    main()
