import functools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from typing import Dict

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
    parser.add_argument('-f', '--histogram-fraction', type=float,
                        help='Fraction of error values used for the histogram', default=1)
    parser.add_argument("-d", "--debugging-file", type=str,
                        help="File name relative to this script where the debugging information will be saved "
                             "(debugging prints will only be saved if this option is included).")
    parser.add_argument('-s', '--save-path', type=str, default='./figures/statistics')
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
    parser.add_argument("--border-area", action='store_true',
                        help="If included, only the South and East error will be calculated with a width of 4 grid cells.")
    args = parser.parse_args()

    Path(args.debugging_file).parent.mkdir(exist_ok=True, parents=True)

    model_class: type[ParentModel] = model_map[args.model[0]]
    with open(args.debugging_file, 'a') as file:
        print(f"\n\n\n[Chosen model is {model_class}]\n", file=file)

    # load model
    model_args = parse_model_args(args.model_args)

    # input normalisation
    df = pd.read_csv(args.input_normalisation_file)
    max_min_list = [(df[df['feature'] == feature]['max'].to_list()[0], df[df['feature'] == feature]['min'].to_list()[0])
                    for feature in args.input_features]

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

    # initialise error_dictionary
    example_point = torch.from_numpy(np.load(test_data[0]))  # could condense but would look horrific
    if model.spatial_reduction == 0:
        cropped_example = example_point
    else:
        cropped_example = example_point[:, :, model.spatial_reduction: -model.spatial_reduction,
                          model.spatial_reduction: -model.spatial_reduction]
    zeros_example = torch.zeros_like(cropped_example[0, 0]).to(device)
    error_dictionary = {'error': torch.zeros(0), 'mae': torch.zeros(1), 'mape': torch.zeros(1), 'mse': torch.zeros(1),
                        'rmse': torch.zeros(1), 'spatial': zeros_example}

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
                            error_dict, length, spatial_length = func(batch_x, batch_y, length, spatial_length)
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
                                error_dict, length, spatial_length = func(batch_x, batch_y, length, spatial_length)
                                file2 = file3
                                break
                            else:
                                input_data = torch.cat([input_file1, input_file2])
                                batch_x, batch_y = preprocessing(input_data)
                                error_dict, length, spatial_length = func(batch_x, batch_y, length, spatial_length)
                                starting_index = batch_start + batch_size - len(file1)
                        elif batch_end - batch_size + model.lead_time <= len(
                                file1):  # if enough data left for >= 1 point, create batch
                            batch_x, batch_y = preprocessing(file1[batch_start:])
                            error_dict, length, spatial_length = func(batch_x, batch_y, length, spatial_length)
                    del file1
                return error_dict

            return wrapper

        return decorator

    @batch(files=test_data, batch_size=args.batch_size, preprocessing=model.preprocess_data)
    def calculate_errors(inputs: torch.Tensor, labels: torch.Tensor, old_len: int, spatial_old_len: int):
        outputs = model(inputs)

        assert len(outputs.shape) == 4
        outputs = outputs[:, error_output_indices]
        labels = labels[:, error_output_indices]
        # if there is a single error output feature, feature dimension might collapse
        if len(outputs.shape) < 4:
            outputs = torch.unsqueeze(outputs, 1)
        if len(labels.shape) < 4:
            labels = torch.unsqueeze(labels, 1)

        outputs = model.swh_reduce(outputs)
        labels = model.swh_reduce(labels)

        spatial = torch.sum(torch.abs(outputs - labels), dim=0)

        spatial_new_len = spatial_old_len + outputs.size(dim=0)

        if args.border_area:
            mask = torch.ones_like(outputs, dtype=torch.bool)
            mask[:-4, :-4] = False
            outputs = outputs[mask]
            labels = labels[mask]

        outputs = outputs[labels != 0]
        labels = labels[labels != 0]

        error: torch.Tensor = outputs - labels
        sample_indices = torch.randperm(error.shape[0])[:int(error.shape[0] * args.histogram_fraction)]
        error = error[sample_indices]
        absolute_error = torch.sum(torch.abs(outputs - labels))
        percentage_error = torch.sum(torch.abs((outputs - labels)) / labels * 100)
        squared_error = torch.sum((outputs - labels) ** 2)

        new_len = old_len + outputs.numel()
        error_dictionary['error'] = torch.cat([error_dictionary['error'], error])
        error_dictionary['mae'] = (error_dictionary['mae'] * old_len + absolute_error) / new_len
        error_dictionary['mape'] = (error_dictionary['mape'] * old_len + percentage_error) / new_len
        error_dictionary['mse'] = (error_dictionary['mse'] * old_len + squared_error) / new_len
        error_dictionary['spatial'] = (error_dictionary['spatial'] * spatial_old_len + spatial) / spatial_new_len
        return error_dictionary, new_len, spatial_new_len

    # determine output features we care about the error for
    error_output_indices = [args.output_features.index(error_output_feature) for error_output_feature in
                            args.error_output_features]

    # calculate errors
    errors = calculate_errors(error_dictionary)
    error_dictionary['rmse'] = error_dictionary['mse'] ** .5
    mask_data = cropped_example[0, -1]
    error_dictionary['spatial'][mask_data == 0] = np.nan

    # print errors to file
    with open(args.debugging_file, 'a') as file:
        print("[Overall Error]\n"
              f"MAE: {float(errors['mae'])}m\n"
              f"MAPE: {float(errors['mape'])}%\n"
              f"MSE: {float(errors['mse'])}m\u00B2\n"
              f"RMSE: {float(errors['rmse'])}m\n", file=file)

    if args.border_area:
        return  # not necessary to code up plotting for only border area

    error_tensor = errors['error'].cpu()
    spatial_tensor = errors['spatial'].cpu()

    # create overall plots
    sns.histplot(error_tensor)
    plt.xlim(-.5, .5)
    plt.ylim(0, 300_000)
    plt.title("Overall Errors from Test Data of 2020-2023")
    plt.xlabel('Prediction minus label (m)')
    plt.ylabel('Frequency')
    save_path = Path(f"{args.save_path}-histogram")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

    plt.xlim(auto=True)
    plt.ylim(auto=True)
    sns.heatmap(spatial_tensor, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Temporally-averaged Overall Error from Test Data of 2020-2023")
    save_path = Path(f"{args.save_path}-spatial-error")
    plt.savefig(save_path)
    plt.clf()

    # save data for making plots
    torch.save(error_tensor, f'{args.debugging_file}-error.pt')
    torch.save(spatial_tensor, f'{args.debugging_file}-spatial.pt')


if __name__ == '__main__':
    main()
