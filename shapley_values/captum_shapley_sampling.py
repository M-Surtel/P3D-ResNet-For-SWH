import argparse
from typing import Dict, Type, Sequence, Iterable
from pathlib import Path
import torch
import os
import time
import pandas as pd
import functools
import random
import captum
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_path = os.path.dirname(current_dir)

from model_classes.abstract_model import ParentModel
from model_classes.unet import UNet
from model_classes.P3D_ResNet import P3DResNet
from model_classes.U_ResNet import UResNet
from model_classes.shapley_P3D_ResNet import ShapleyP3DResNet

from training.train import parse_model_args, get_dataset_files


def constrained_randperm(num_features: int, ancestors: int | Sequence[int], descendants: int | Sequence[int]) -> torch.Tensor:
    if type(ancestors) == int:
        ancestors = (ancestors,)
    if type(descendants) == int:
        descendants = (descendants,)
    if any(index >= num_features for index in ancestors + descendants):
        raise ValueError(
            '[constrained_randperm]: at least one of the ancestors/descendants indices is out of bounds (>= num_features)')
    ancestors = torch.tensor(ancestors)
    descendants = torch.tensor(descendants)

    all_features = torch.arange(num_features)
    ancestor_mask = torch.isin(all_features, ancestors)
    descendant_mask = torch.isin(all_features, descendants)
    other_features = all_features[~(ancestor_mask + descendant_mask)]

    other_indices = torch.randperm(len(other_features))
    shuffled_others = other_features[other_indices]

    ancestor_indices = torch.randperm(len(ancestors))
    shuffled_ancestors = ancestors[ancestor_indices]

    descendant_indices = torch.randperm(len(descendants))
    shuffled_descendants = descendants[descendant_indices]

    anc_n_desc_indices = sorted(random.sample(tuple(f for f in range(num_features)), len(ancestors) + len(descendants)))
    result = torch.full((num_features,), -1, dtype=torch.long)
    result[anc_n_desc_indices] = torch.cat([shuffled_ancestors, shuffled_descendants])
    result[result == -1] = shuffled_others
    return result

def _time_step_asv_perm_generator(num_time_steps: int, num_samples: int) -> Iterable[Sequence[int]]:
    for _ in range(num_samples):
        yield torch.arange(num_time_steps).tolist()

def parse_user_args():
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Arguments related to the model')
    model_group.add_argument(
        "-m", "--model",
        metavar="MODEL",
        nargs=1,
        required=True,
        help="(Required) Choose which model to train.",
        choices=model_and_on_manifold_to_class.keys()
    )
    model_group.add_argument(
        "-a", "--model-args",
        nargs='+',
        help=f"Provide space-separated arguments for the initialisation of the chosen model."
             f"\nOnly supply arguments for the chosen model class.",
    )
    model_group.add_argument('-c', '--checkpoint', type=str, help='Model parameters to load.', required=True)
    data_group = parser.add_argument_group('Arguments related to the data')
    data_group.add_argument("-n", "--input-normalisation-file", type=str,
                            help="(Required) min-max-mean input normalisation is applied using the min-max value provided in file."
                                 "Input normalisation file is needed for determining indices of input/output variables."
                                 "Feature means are used to mask out-of-coalition features in off-manifold Shapley analysis.")
    data_group.add_argument('-i', '--input-features', type=str, nargs='+', required=True,
                            help="(Required) Names of chosen input variables as they appear in the user-supplied normalisation file.")
    data_group.add_argument('-u', '--output-features', type=str, nargs='+', required=True,
                            help="(Required) Names of chosen output variables as they appear in the user-supplied normalisation file.")
    data_group.add_argument('--ancestor-features', type=str, nargs='+', required=False,
                            help="(Optional) Names of chosen ancestor variables as they appear in the user-supplied normalisation file.")
    data_group.add_argument('--descendant-features', type=str, nargs='+', required=False,
                            help="(Optional) Names of chosen ancestor variables as they appear in the user-supplied normalisation file.")
    data_group.add_argument("-t", "--testing-data", type=str, nargs=1, required=True,
                            help="(Required) Path relative to this script to the dataset file."
                                 "\n If the filename conforms to the regular expression '*(0).npy', "
                                 "this script will automatically try to find files of '[Name](N).npy' "
                                 "until it reaches an N for which there is no file.")
    data_group.add_argument("-s", "--sea-mask", type=str,
                            help="(Required) Path to .npy file that contains a Boolean tensor with True for sea "
                                 "and False for land. Must be same shape as area.", required=True)
    data_group.add_argument("-b", "--batch-size", type=int, help="Size of each batch (Default is 1)", default=1)
    data_group.add_argument("-e", "--num-samples", type=int, help="The number of feature permutations tested (Default is 25)", default=25)
    data_group.add_argument("-p", "--perturbations-per-eval", type=int,
                            help="Allows multiple ablations to be processed simultaneously in one call to forward_fn. "
                                 "Each forward pass will contain a maximum of perturbations_per_eval * #examples samples. (Default is 1)", default=1)
    data_group.add_argument("--on-manifold", action='store_true',
                            help="When included, the feature_mask will be adjusted to be on-manifold and the on-manifold model class will be used")
    data_group.add_argument("--accuracy-based", action='store_true',
                            help="When included, Shapley values will be calculated based on prediction accuracy rather than prediction magnitude.")
    data_group.add_argument("--time-step-asv", action='store_true',
                            help="When included, ASVs will be calculated of the time steps rather than the input features."
                                 "This setting overrides '--perturbations-per-eval' and sets it to one, as the only order "
                                 "for temporal ASVs is the order of the time steps")
    output_group = parser.add_argument_group('Arguments related to the output')
    output_group.add_argument("-d", "--debugging-file", type=str, required=True,
                              help="File name relative to this script where the debugging information will be saved.")
    output_group.add_argument("-o", "--output-file", type=str, required=True,
                              help="File name relative to this script where the output tensor providing analysis for each batch will be saved ")
    return parser.parse_args()


def debug_print(string, debugging_file):
    if debugging_file:
        with open(debugging_file, 'a') as file:
            print(string, file=file)


# TODO: fix input_sea_mask for model.crop_input == True
def create_wrapper_class(base_class: Type[ParentModel], input_sea_mask: torch.Tensor, model_args: Dict, is_accuracy_based: bool):
    ## calculate output_sea_mask
    if 'spatial_reduction' in model_args:
        spatial_reduction = model_args['spatial_reduction'] // 2
    else:
        spatial_reduction = 0
    output_sea_mask: torch.Tensor = input_sea_mask[spatial_reduction: -spatial_reduction, spatial_reduction: -spatial_reduction]
    output_sea_elements = torch.sum(output_sea_mask)

    class MagnitudeBasedWrapper(base_class):
        __name__ = f"Wrapped{base_class.__name__}"

        def forward(self, inputs: torch.Tensor):
            output = super().forward(inputs)
            expanded_sea_mask = output_sea_mask.expand(output.shape[0], output.shape[1], -1, -1)
            result = torch.empty((output.shape[0], output_sea_elements))
            for n in range(output.shape[0]):
                result[n] = output[n][expanded_sea_mask[0]]
            return result.sum(dim=1)

    class AccuracyBasedWrapper(base_class):
        __name__ = f"Wrapped{base_class.__name__}"

        def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
            output = super().forward(inputs)
            expanded_sea_mask = output_sea_mask.expand(output.shape[0], output.shape[1], -1, -1)
            result = torch.empty((output.shape[0], output_sea_elements))
            for n in range(output.shape[0]):
                error = torch.abs(output[n][expanded_sea_mask[0]] - labels[n][expanded_sea_mask[0]])
                accuracy = - torch.log10(error + 10e-7)  # add small number to avoid running torch.log10(0)
                result[n] = accuracy
            return result.sum(dim=1)

    return AccuracyBasedWrapper if is_accuracy_based else MagnitudeBasedWrapper


def print_progress_update(total_attributions, debugging_file, input_features, output_file, batch_counter=None, start_time=None):
    mean_attributions = torch.mean(total_attributions, 0)
    if start_time is None:
        debug_print('\n\nFinal results:', debugging_file)
    else:
        debug_print(f'\nIntermittent Results after Batch {batch_counter} (completion time {time.time() - start_time}s):', debugging_file)
    for feature_index, feature_name in enumerate(input_features):
        debug_print(f'{feature_name:<{7}}\t\t\t'
                    f'{float(mean_attributions[feature_index]):<.2e}', debugging_file)

    # save output
    if batch_counter % 100:
        output_dict = {'attributions': total_attributions, 'input features': input_features}
        torch.save(output_dict, output_file)


def print_progress_update_time_step_asv(total_attributions, debugging_file, output_file, time_steps, batch_counter=None, start_time=None):
    mean_attributions = torch.mean(total_attributions, 0)
    if start_time is None:
        debug_print('\n\nFinal results:', debugging_file)
    else:
        debug_print(f'\nIntermittent Results after Batch {batch_counter} (completion time {time.time() - start_time}s):', debugging_file)
    for time_step in range(time_steps):
        debug_print(f'Time step {time_step:<{3}}\t\t\t'
                    f'{float(mean_attributions[time_step]):<.2e}', debugging_file)

    # save output
    if batch_counter is None or batch_counter % 100:
        output_dict = {'attributions': total_attributions, 'time steps': time_steps}
        torch.save(output_dict, output_file)



def calculate_shapley(args, data_files, model, feature_to_mean, sea_mask, device):
    def batch(files, batch_size, input_features, lookback, lead_time, preprocessing, debugging_file, output_file):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*vargs, **kwargs):
                if args.time_step_asv:
                    total_attributions: torch.Tensor = torch.empty((0, model.lookback + 1))
                else:
                    total_attributions: torch.Tensor = torch.empty((0, len(input_features)))
                input_file_counter = 0
                batch_counter = 0

                file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device) if input_file_counter < len(
                    files) else None
                input_file_counter += 1

                starting_index = 0  # a file's 1st batch doesn't often start at the file's 1st index
                while file2 is not None:
                    file1 = file2
                    file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device) if input_file_counter < len(
                        files) else None
                    input_file_counter += 1

                    # file1 batches
                    for batch_start in range(starting_index, len(file1), batch_size):
                        start_time = time.time()
                        batch_x: torch.Tensor
                        batch_end = batch_start + batch_size + lookback + lead_time
                        # batches using 1 file
                        if batch_end <= len(file1):
                            batch_x, batch_y = preprocessing(file1[batch_start:batch_end])
                            batch_counter += 1
                            batch_attributions = func(batch_x, batch_y)
                            total_attributions = torch.cat((total_attributions, torch.unsqueeze(batch_attributions, 0)))
                        # batches using 2 files
                        elif file2 is not None:
                            input_file1 = file1[batch_start:]
                            input_file2 = file2[:batch_end - len(file1)]
                            if len(input_file1) + len(
                                    input_file2) < batch_size + model.lookback + 1 and input_file_counter < len(files):
                                file3 = torch.from_numpy(np.load(files[input_file_counter])).to(device)
                                input_file_counter += 1
                                input_file3 = file3[:batch_end - len(file1) - len(file2)]
                                input_data = torch.cat([input_file1, input_file2, input_file3])
                                batch_x, batch_y = preprocessing(input_data)
                                batch_counter += 1
                                batch_attributions = func(batch_x, batch_y)
                                total_attributions = torch.cat((total_attributions, torch.unsqueeze(batch_attributions, 0)))
                                file2 = file3
                                break
                            else:
                                input_data = torch.cat([input_file1, input_file2])
                                batch_x, batch_y = preprocessing(input_data)
                                batch_counter += 1
                                batch_attributions = func(batch_x, batch_y)
                                total_attributions = torch.cat((total_attributions, torch.unsqueeze(batch_attributions, 0)))
                        elif batch_end - batch_size + 1 <= len(file1):  # if enough data left for >= 1 point, create batch
                            batch_x, batch_y = preprocessing(file1[batch_start:])
                            batch_counter += 1
                            batch_attributions = func(batch_x, batch_y)
                            total_attributions = torch.cat((total_attributions, torch.unsqueeze(batch_attributions, 0)))

                        # progress update
                        if args.time_step_asv:
                            print_progress_update_time_step_asv(total_attributions, debugging_file, output_file, model.lookback + 1, batch_counter, start_time)
                        else:
                            print_progress_update(total_attributions, debugging_file, input_features, output_file, batch_counter, start_time)
                    del file1
                return total_attributions
            return wrapper
        return decorator

    @batch(data_files, args.batch_size, args.input_features, model.lookback, model.lead_time, model.preprocess_data, args.debugging_file, args.output_file)
    def sample_shapley(inputs: torch.Tensor, labels: torch.Tensor):
        attributions = shapley_sampling.attribute(
            inputs=inputs,
            baselines=baseline,
            feature_mask=feature_mask,
            n_samples=args.num_samples,
            perturbations_per_eval=args.perturbations_per_eval,
            additional_forward_args=labels if args.accuracy_based else None
        )
        # attributions shape is (N, F_I, T_I, Lat_I, Lon_I)
        # output part of attributions' shape is gone because our wrapper reduces outputs to be of shape (N,)
        if args.time_step_asv:
            attributions = torch.mean(attributions, dim=(0, 1, 3, 4))  # only retain T_I
        else:
            attributions = torch.mean(attributions, dim=(0, 2, 3, 4))  # only retain F_I
        return attributions

    # data preparation
    feature_mask = torch.empty((1, len(args.input_features), model.lookback + 1, *sea_mask.shape))  # data shape is (N, F, T, Lat, Lon)
    if args.time_step_asv:
        # the features analysed by Shapley values are the time steps
        for time_step_index in range(model.lookback + 1):
            # invert so that most recent time step is 0 and oldest time step is model.lookback,
            # as the question is 'how far back do we need to expand the temporal window size?'
            feature_mask[:, :, time_step_index] = model.lookback - time_step_index
    else:
        # the features analysed by Shapley values are the input features
        for feature_index in range(len(args.input_features)):
            feature_mask[:, feature_index] = feature_index
    baseline = torch.zeros((1, len(args.input_features), model.lookback + 1, *sea_mask.shape))  # data shape is (N, F, T, Lat, Lon)
    for feature_index, feature_name in enumerate(args.input_features):
        if args.on_manifold:
            # baseline[:, feature_index] = torch.where(sea_mask, float('nan'), 0)
            # use -99,999 as substitute for NaN; NaN messes with the tensor operations
            baseline[:, feature_index] = torch.where(sea_mask, -99_999, 0)
        else:
            baseline[:, feature_index] = torch.where(sea_mask, feature_to_mean[feature_name], 0)

    # perform Shapley sampling
    shapley_sampling = captum.attr.ShapleyValueSampling(model)
    if args.ancestor_features is not None and args.descendant_features is not None:
        # define modified permutation generator such that descendant variables never appear before ancestor variables (for ASVs)
        ancestor_indices = [args.input_features.index(ancestor_feature) for ancestor_feature in args.ancestor_features]
        descendant_indices = [args.input_features.index(descendant_feature) for descendant_feature in args.descendant_features]
        def _perm_generator_asv(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
            for _ in range(num_samples):
                yield constrained_randperm(num_features, ancestors=ancestor_indices, descendants=descendant_indices).tolist()
        shapley_sampling.permutation_generator = _perm_generator_asv
    if args.time_step_asv:
        # replace with a perm gen that just returns torch.arange
        shapley_sampling.permutation_generator = _time_step_asv_perm_generator
        args.perturbations_per_eval = 1  # there is only one ordering for time steps, so multiple perturbations make no sense
    total_attributions: torch.Tensor = sample_shapley()
    mean_attributions = torch.mean(total_attributions, 0)

    # print & save output
    if args.time_step_asv:
        print_progress_update_time_step_asv(total_attributions, args.debugging_file, args.output_file, model.lookback + 1)
    else:
        print_progress_update(total_attributions, args.debugging_file, args.input_features, args.output_file)


def main():
    args = parse_user_args()
    if args.debugging_file:
        Path(args.debugging_file).parent.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        debug_print("CUDA is available!  Training on GPU...", args.debugging_file)
        device = torch.device('cuda')
    else:
        debug_print("CUDA not available, training on CPU...", args.debugging_file)
        device = torch.device('cpu')
    torch.set_default_device(device)

    sea_mask = torch.from_numpy(np.load(args.sea_mask))

    model_class: type[ParentModel] = model_and_on_manifold_to_class[args.model[0]][args.on_manifold]
    model_args = parse_model_args(args.model_args)
    wrapped_model_class = create_wrapper_class(model_class, sea_mask, model_args, args.accuracy_based)
    debug_print("[main]: parse_model_args()", args.debugging_file)

    # input normalisation
    df = pd.read_csv(args.input_normalisation_file)
    max_min_list = [(df[df['feature'] == feature]['max'].to_list()[0], df[df['feature'] == feature]['min'].to_list()[0]) for feature in args.input_features]
    feature_to_mean: Dict[str, float] = {df['feature'][i]: df['mean'][i] for i in range(len(df))}

    debug_print("[main]: instantiate model", args.debugging_file)
    if torch.cuda.is_available():
        model: ParentModel = wrapped_model_class(df['feature'].to_list(), args.input_features, args.output_features,
                                         **model_args, max_min_list=max_min_list).cuda()
    else:
        model: ParentModel = wrapped_model_class(df['feature'].to_list(), args.input_features, args.output_features,
                                         **model_args, max_min_list=max_min_list)

    debug_print("[main]: load checkpoint", args.debugging_file)
    checkpoint: Dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.train(False)

    dataset_files = get_dataset_files(args.testing_data[0])

    calculate_shapley(args, dataset_files, model, feature_to_mean, sea_mask, device)

model_and_on_manifold_to_class: Dict[str, Dict[bool, type[ParentModel]]] = {
    'U-Net': {False: UNet, True: UNet},
    'P3D-ResNet': {False: P3DResNet, True: ShapleyP3DResNet},
    'U-ResNet': {False: UResNet, True: UResNet},
}
if __name__ == '__main__':
    main()
