import functools
import torch
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from typing import Dict

from model_classes.abstract_model import ParentModel
from model_classes.persistence_model import PersistenceModel
from model_classes.P3D_ResNet import P3DResNet
from training.train import parse_model_args
from training.train import get_dataset_files


def user_args(model_map):
    parser = argparse.ArgumentParser()

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
    parser.add_argument('-r', '--respective-feature', type=str, nargs=1, required=True,
                        help="(Required) Name of chosen variable that should be on the x-axis of the produced graph "
                             "(i.e. error with respect to '-r'), as it appears in the user-supplied normalisation file.")
    return parser.parse_args()


def main():
    #############################################
    #                   SETUP
    #############################################
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
    }
    args = user_args(model_map)

    torch.set_grad_enabled(False)

    Path(args.debugging_file).parent.mkdir(exist_ok=True, parents=True)

    model_class: type[ParentModel] = model_map[args.model[0]]
    with open(args.debugging_file, 'a') as file:
        print(f"\n\n\n[Chosen model is {model_class}]\n", file=file)

    # load model
    model_args = parse_model_args(args.model_args)

    # input normalisation
    df = pd.read_csv(args.input_normalisation_file)
    max_min_list = [(df[df['feature'] == feature]['max'].to_list()[0],
                     df[df['feature'] == feature]['min'].to_list()[0])
                    for feature in args.input_features]

    with open(args.debugging_file, 'a') as file:
        print("[main]: instantiate model, loss, optimiser", file=file)
    if torch.cuda.is_available():
        model: ParentModel = model_class(
            df['feature'].to_list(),
            args.input_features,
            args.output_features,
            **model_args,
            max_min_list=max_min_list,
            additional_output_features=args.respective_feature
        ).cuda()
    else:
        model: ParentModel = model_class(
            df['feature'].to_list(),
            args.input_features,
            args.output_features,
            **model_args,
            max_min_list=max_min_list,
            additional_output_features=args.respective_feature
        )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # load data
    test_data = get_dataset_files(args.data[0])

    #############################################
    #          CALCULATING RESULTS
    #############################################
    def batch(files, batch_size, preprocessing):
        def decorator(func):
            @functools.wraps(func)
            def wrapper():
                window_size = batch_size + model.lookback + model.lead_time  # The total raw data points needed to create one full batch
                error_tensor = torch.zeros(0)
                feature_tensor = torch.zeros(0)  # tensor for holding --respective-feature values
                buffer = None
                for file_path in files:
                    new_data = torch.from_numpy(np.load(file_path)).to(device)

                    if buffer is None:
                        buffer = new_data
                    else:
                        # this only runs when the buffer did not have enough data to create a batch
                        # which means it loads in a new file and appends the data to the buffer here
                        buffer = torch.cat((buffer, new_data), dim=0)

                    while len(buffer) >= window_size:
                        raw_window = buffer[:window_size]
                        # batch_z is the --respective-feature
                        batch_x, batch_y, batch_z = preprocessing(raw_window)
                        batch_z = batch_z.squeeze()  # squeeze feature dimension, as there will only be 1 feature
                        new_error, new_feature = func(batch_x, batch_y, batch_z)
                        error_tensor = torch.cat([error_tensor, new_error])
                        feature_tensor = torch.cat([feature_tensor, new_feature])
                        buffer = buffer[batch_size:]  # slide buffer forward
                if buffer is not None and len(buffer) > model.lookback + model.lead_time:  # enough data to create at least one more data point
                    batch_x, batch_y, batch_z = preprocessing(buffer)
                    batch_z = batch_z.squeeze()  # squeeze feature dimension, as there will only be 1 feature
                    new_error, new_feature = func(batch_x, batch_y, batch_z)
                    error_tensor = torch.cat([error_tensor, new_error])
                    feature_tensor = torch.cat([feature_tensor, new_feature])
                return error_tensor, feature_tensor
            return wrapper
        return decorator

    @batch(files=test_data, batch_size=args.batch_size, preprocessing=model.preprocess_data)
    def calculate_errors(inputs: torch.Tensor, labels: torch.Tensor, feature: torch.Tensor):
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

        # ensure no 0-values
        outputs = outputs[labels != 0]
        feature = feature[labels != 0]
        labels = labels[labels != 0]

        error: torch.Tensor = outputs - labels
        sample_indices = torch.randperm(error.shape[0])[:int(error.shape[0] * args.histogram_fraction)]
        error = error[sample_indices]
        feature = feature[sample_indices]
        return error, feature

    error_output_indices = [args.output_features.index(error_output_feature) for error_output_feature in
                            args.error_output_features]

    error_tensor, feature_tensor = calculate_errors()

    #############################################
    #             SAVING RESULTS
    #############################################
    error_tensor = error_tensor.cpu()
    feature_tensor = feature_tensor.cpu()

    Path(f'{args.debugging_file}-values.pt').unlink(missing_ok=True)
    torch.save({'error': error_tensor, args.respective_feature[0]: feature_tensor}, f'{args.debugging_file}-values.pt')


if __name__ == '__main__':
    main()
