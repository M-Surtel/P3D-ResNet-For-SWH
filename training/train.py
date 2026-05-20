import argparse
import sys
from typing import Dict, Union
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.utils import clip_grad_norm_
import time
import natsort
import pandas as pd
import functools
import inspect

import numpy as np


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_path = os.path.dirname(current_dir)

from model_classes.abstract_model import ParentModel
from model_classes.unet import UNet
from model_classes.P3D_ResNet import P3DResNet
from model_classes.U_ResNet import UResNet


class WeightDecayScheduler:
    def __init__(self, optimiser: torch.optim.Optimizer, decrease_factor: float, frequency: int):
        self.optimiser = optimiser
        self.factor = decrease_factor
        self.frequency = frequency
        self.counter = 0

    def step(self):
        self.counter += 1
        if self.counter % 3 == 0:
            for param_group in self.optimiser.param_groups:
                param_group['weight_decay'] *= self.factor


def parse_user_args():
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Arguments related to the model')
    model_group.add_argument(
        "-m", "--model",
        metavar="MODEL",
        nargs=1,
        required=True,
        help="(Required) Choose which model to train.",
        choices=model_map.keys()
    )
    model_group.add_argument(
        "-a", "--model-args",
        nargs='+',
        help=f"Provide space-separated arguments for the initialisation of the chosen model."
             f"\nOnly supply arguments for the chosen model class.",
    )
    model_group.add_argument('-c', '--checkpoint', type=str, help='If included, start training from a certain checkpoint instead of from scratch.')
    training_group = parser.add_argument_group('Arguments related to the training')
    training_group.add_argument("-n", "--input-normalisation-file", type=str,
                             help="Min-max input normalisation is applied using the min-max value provided in file."
                                  "Input normalisation file is needed for determining indices of input/output variables."
                                  "Should probably implement way to run this script without max/min normalisation."
                                  "If not included, it is assumed that there is only one feature.")
    training_group.add_argument('-i', '--input-features', type=str, nargs='+',
                                help="Names of chosen input variables as they appear in the user-supplied normalisation file."
                                     "If not included, it is assumed that there is only one feature.")
    training_group.add_argument('-u', '--output-features', type=str, nargs='+',
                                help="Names of chosen output variables as they appear in the user-supplied normalisation file. "
                                     "If not included, it is assumed that there is only one feature.")
    training_group.add_argument("-t", "--training-data", type=str, nargs=1, required=True,
                                help="(Required) Path relative to this script to the dataset file."
                                     "\n If the filename conforms to the regular expression '*(0).npy', "
                                     "this script will automatically try to find files of '[Name](N).npy' "
                                     "until it reaches an N for which there is no file.")
    training_group.add_argument("-e", "--epochs", type=int, help="(Required) Number of epochs to train", required=True)
    training_group.add_argument("-b", "--batch-size", type=int, help="Size of each training batch (Default is 1)", default=1)
    training_group.add_argument("-g", "--gradient-accumulation", type=int, help="Number of batches before gradients get applied (Default is 1)", default=1)
    training_group.add_argument("-v", "--validation-data", type=str, nargs=1, help="Path to validation dataset")
    training_group.add_argument("-p", "--precision", default='float32', choices=['float32', 'bfloat16', 'mixed'],
                        help="Precision of the training/network. Note bfloat16 causes model params to be bfloat16, while mixed and float32 come with float32 model parameters. Default is float32.")
    training_group.add_argument("-f", "--weight-decay-decrease-factor", type=float, help="The factor by which the weight decay is decreased every epoch."
                                                                                         "If left unspecified weight decay does not get decreased.")
    training_group.add_argument("-s", "--weight-decay-step-size", type=int, help="The number of epochs before an update occurs to weight decay.", default=1)
    output_group = parser.add_argument_group('Arguments related to the output')
    output_group.add_argument("-o", "--output", type=str, default="./model_parameters/unnamed-model",
                              help="Path relative to this script where the model parameters should be saved "
                                   "(default = ./model_parameters/unnamed-model).")
    output_group.add_argument("-l", "--loss-graph", type=str,
                              help="File name relative to this script where the loss graph will be saved "
                                   "(loss graph will only be saved if this option is included).")
    output_group.add_argument("-d", "--debugging-file", type=str,
                              help="File name relative to this script where the debugging information will be saved "
                                   "(debugging prints will only be saved if this option is included).")
    return parser.parse_args()


def parse_model_args(args: list) -> Dict[str, Union[int, float, str]]:
    model_args: Dict[str, Union[int, float, str]] = {}
    for arg in args:
        key, value = arg.split("=")
        try:
            model_args[key] = eval(value)
        except NameError:
            model_args[key] = value  # Treat as string if conversion fails
    return model_args

def conditional_autocast(func):
    """
    Wraps the function func such that func gets @torch.autocasted if an argument named 'arg' (i.e. argparser namespace)
    contains an entry for 'precision' == 'mixed'.  This allows the training loop to be a single function for multiple
    machine precisions.

    :param func: function to be wrapped
    :return: wrapper object
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            args_obj = bound_args.arguments.get('args')
        except TypeError as e:
            print(f'[ERROR][Conditional Autocast]: Could not find the args object. Running without autocast. Error: {e}')
            args_obj = None

        if args_obj and hasattr(args_obj, 'precision') and args_obj.precision == 'mixed':
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[Conditional Autocast]: Applying torch.autocast(device_type='{device_type}')")
            with torch.autocast(device_type=device_type):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


@conditional_autocast
def training_loop(model, loss_function, optimiser, scheduler, weight_decay_scheduler, training_files, validation_files, args, device, starting_epoch,
                          train_losses, train_swh_losses, validation_losses, validation_swh_losses) -> tuple[list[float], list[float]]:
    if args.precision == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    def batch(files, batch_size, lookback, lead_time, preprocessing, is_training):
        def decorator(func):
            @functools.wraps(func)
            @torch.set_grad_enabled(is_training)
            def wrapper(*args, **kwargs):
                epoch_loss = 0.0
                epoch_swh_loss = 0.0
                input_file_counter = 0
                batch_counter = 0

                file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device, dtype=dtype) if input_file_counter < len(files) else None
                input_file_counter += 1

                starting_index = 0  # a file's 1st batch doesn't often start at the file's 1st index
                while file2 is not None:
                    file1 = file2
                    file2 = torch.from_numpy(np.load(files[input_file_counter])).to(device, dtype=dtype) if input_file_counter < len(files) else None
                    input_file_counter += 1

                    # file1 batches
                    for batch_start in range(starting_index, len(file1), batch_size):
                        batch_x: torch.Tensor
                        batch_y: torch.Tensor
                        batch_end = batch_start + batch_size + lookback + lead_time
                        # batches using 1 file
                        if batch_end <= len(file1):
                            batch_x, batch_y = preprocessing(file1[batch_start:batch_end])
                            batch_counter += 1
                            step_loss, swh_loss = func(batch_x, batch_y, batch_counter)
                            epoch_loss += step_loss
                            epoch_swh_loss += swh_loss
                        # batches using 2 files
                        elif file2 is not None:
                            input_file1 = file1[batch_start:]
                            input_file2 = file2[:batch_end - len(file1)]
                            if len(input_file1) + len(input_file2) < batch_size + lookback + lead_time and input_file_counter < len(files):
                                file3 = torch.from_numpy(np.load(files[input_file_counter])).to(device, dtype=dtype)
                                input_file_counter += 1
                                input_file3 = file3[:batch_end - len(file1) - len(file2)]
                                input_data = torch.cat([input_file1, input_file2, input_file3])
                                batch_x, batch_y = preprocessing(input_data)
                                batch_counter += 1
                                step_loss, swh_loss = func(batch_x, batch_y, batch_counter)
                                epoch_loss += step_loss
                                epoch_swh_loss += swh_loss
                                file2 = file3
                                break
                            else:
                                input_data = torch.cat([input_file1, input_file2])
                                batch_x, batch_y = preprocessing(input_data)
                                batch_counter += 1
                                step_loss, swh_loss = func(batch_x, batch_y, batch_counter)
                                starting_index = batch_start + batch_size - len(file1)
                                epoch_loss += step_loss
                                epoch_swh_loss += swh_loss
                        elif batch_end - batch_size + lead_time <= len(file1):  # if enough data left for >= 1 point, create batch
                            batch_x, batch_y = preprocessing(file1[batch_start:])
                            batch_counter += 1
                            step_loss, swh_loss = func(batch_x, batch_y, batch_counter)
                            epoch_loss += step_loss
                            epoch_swh_loss += swh_loss
                    del file1
                avg_epoch_loss = epoch_loss / batch_counter
                avg_epoch_swh_loss = epoch_swh_loss / batch_counter
                return avg_epoch_loss, avg_epoch_swh_loss
            return wrapper
        return decorator

    @batch(files=training_files, batch_size=args.batch_size, lookback=model.lookback, lead_time=model.lead_time, preprocessing=model.preprocess_data, is_training=True)
    def train_step(inputs: torch.Tensor, labels: torch.Tensor, batch_counter: int):
        outputs = model(inputs.detach())
        loss = loss_function(outputs[labels != 0], labels[labels != 0])
        outputs = model.swh_reduce(outputs)
        labels = model.swh_reduce(labels)
        swh_loss = loss_function(outputs[labels != 0], labels[labels != 0])
        loss.backward()
        if batch_counter % args.gradient_accumulation == 0:  # Only optimise each Nth step (user-specified)
            try:
                clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)  # prevent infs and NaNs
            except RuntimeError as e:
                # NaN loss occurred
                print(e)
                sys.exit(2)  # let Bayesian optimisation script know error was due to NaN loss
            optimiser.step()
            optimiser.zero_grad()
        return loss.item(), swh_loss.item()

    @batch(files=validation_files, batch_size=args.batch_size, lookback=model.lookback, lead_time=model.lead_time, preprocessing=model.preprocess_data, is_training=False)
    def validation_step(inputs: torch.Tensor, labels: torch.Tensor, batch_counter: int):  # batch_counter unused here but still supplied because of @batch
        outputs = model(inputs)
        loss = loss_function(outputs[labels != 0], labels[labels != 0])
        outputs = model.swh_reduce(outputs)
        labels = model.swh_reduce(labels)
        swh_loss = loss_function(outputs[labels != 0], labels[labels != 0])
        return loss.item(), swh_loss.item()

    if scheduler is not None and 'metrics' in scheduler.step.__code__.co_varnames and validation_files is None:
        raise Exception("In order to use torch.optim.lr_scheduler.ReduceLROnPlateau, you need validation loss.")

    best_training_swh_loss = min(train_losses) if len(train_losses) > 0 else float('inf')
    best_validation_swh_loss = min(validation_losses) if len(validation_losses) > 0 else float('inf')
    for epoch in range(starting_epoch, args.epochs):
        start_time = time.time()
        training_loss, training_swh_loss = train_step()
        optimiser.step()
        optimiser.zero_grad()
        train_losses.append(training_loss)
        train_swh_losses.append(training_swh_loss)

        # note validation is always tested AFTER training's backprop (so validation performance should be better than training)
        if validation_files is not None:
            validation_loss, validation_swh_loss = validation_step()
            validation_losses.append(validation_loss)
            validation_swh_losses.append(validation_swh_loss)

        if scheduler is not None:
            if 'metrics' in scheduler.step.__code__.co_varnames:
                scheduler.step(validation_loss)  # ReduceLROnPlateau requires a metric for step()
            else:
                scheduler.step()  # No other scheduler requires parameters for step()

        if weight_decay_scheduler:
            weight_decay_scheduler.step()

        with open(args.debugging_file, 'a') as file:
            print(f"[training_loop]: Completed epoch {epoch+1}/{args.epochs} in {time.time()-start_time:.0f} seconds,\n"
                  f" training losses are: {train_losses}\n"
                  f" training swh losses are: {train_swh_losses}\n"
                  f" validation losses are: {validation_losses if validation_files is not None else torch.nan}\n"
                  f" validation swh losses are: {validation_swh_losses if validation_files is not None else torch.nan}", file=file)
        if model.scheduler is None:
            scheduler_state_dict = None
        else:
            scheduler_state_dict = model.scheduler.state_dict()
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimiser': model.optimiser.state_dict(),
            'lr_scheduler': scheduler_state_dict,
            'training_losses': train_losses,
            'training_swh_losses': train_swh_losses,
            'validation_losses': validation_losses,
            'validation_swh_losses': validation_swh_losses,
        }
        torch.save(checkpoint, args.output if str(args.output).endswith('.pth') else args.output + '.pth')
        if training_swh_loss < best_training_swh_loss:
            best_training_swh_loss = training_swh_loss
            torch.save(checkpoint,  args.output + '_best_training_swh_loss.pth')
        if validation_files is not None:
            if validation_swh_loss < best_validation_swh_loss:
                best_validation_swh_loss = validation_swh_loss
                torch.save(checkpoint,  args.output + '_best_validation_swh_loss.pth')
    return train_losses, validation_losses


def plot_and_save_losses(training_losses, validation_losses, save_path):
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")

    sns.lineplot(x=epochs, y=training_losses, label="Training Loss")
    if len(validation_losses) > 0:
        sns.lineplot(x=epochs, y=validation_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.yscale('log')  # exponential scaling
    plt.legend()

    save_path = Path(save_path if str(save_path).endswith('.png') else save_path + '.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)


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
    args = parse_user_args()

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    Path(args.debugging_file).parent.mkdir(exist_ok=True, parents=True)

    with open(args.debugging_file, 'a') as file:
        if torch.cuda.is_available():
            print("CUDA is available!  Training on GPU...", file=file)
            device = torch.device('cuda')
        else:
            print("CUDA not available, training on CPU...", file=file)
            device = torch.device('cpu')
    torch.set_default_device(device)

    model_class: type[ParentModel] = model_map.get(args.model[0])  # Argparse choices ensure valid key, preventing None type
    with open(args.debugging_file, 'a') as file:
        print("[main]: parse_model_args()", file=file)
    model_args = parse_model_args(args.model_args)

    if args.input_normalisation_file:
        # input normalisation
        df = pd.read_csv(args.input_normalisation_file)
        model_args['max_min_list'] = [(df[df['feature'] == feature]['max'].to_list()[0], df[df['feature'] == feature]['min'].to_list()[0]) for feature in args.input_features]
        model_args['feature_list'] = df['feature'].to_list()
    if args.input_features:
        model_args['input_features'] = args.input_features
    if args.output_features:
        model_args['output_features'] = args.output_features


    with open(args.debugging_file, 'a') as file:
        print("[main]: instantiate model, loss, optimiser", file=file)
    if torch.cuda.is_available():
        model: ParentModel = model_class(**model_args).cuda()
    else:
        model: ParentModel = model_class(**model_args)
    if args.precision == 'bfloat16':
        model = model.to(dtype=torch.bfloat16)

    checkpoint: Dict | None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    else:
        checkpoint = None

    loss_function = model.loss_function
    optimiser = model.optimiser

    scheduler = model.scheduler
    # TODO: even with the fancy saving tech weight_decay_scheduler is unaccounted for (mostly unused anyways but still)
    weight_decay_scheduler = WeightDecayScheduler(optimiser, args.weight_decay_decrease_factor, args.weight_decay_step_size) \
        if args.weight_decay_decrease_factor else None

    starting_epoch: int = 0
    training_losses: list = []
    training_swh_losses: list = []
    validation_losses: list = []
    validation_swh_losses: list = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        starting_epoch = checkpoint['epoch']
        training_losses = checkpoint['training_losses']
        training_swh_losses = checkpoint['training_swh_losses']
        validation_losses = checkpoint['validation_losses']
        validation_swh_losses = checkpoint['validation_swh_losses']

    training_files = get_dataset_files(args.training_data[0])
    if args.validation_data is not None:
        validation_files = get_dataset_files(args.validation_data[0])
    else:
        validation_files = None

    with open(args.debugging_file, 'a') as file:
        print("[main]: training_loop()", file=file)
    training_loop(model, loss_function, optimiser, scheduler, weight_decay_scheduler, training_files, validation_files,
                  args, device, starting_epoch, training_losses, training_swh_losses, validation_losses, validation_swh_losses)

    if args.loss_graph is not None:
        plot_and_save_losses(training_losses, validation_losses, args.loss_graph)

    with open(args.debugging_file + '_results', 'a') as file:
        print(f"training loss {training_losses}, validation loss {validation_losses}", file=file)
        print(f"parameters: {model_args}", file=file)

    if validation_files is not None:
        print(min(validation_swh_losses), end='')  # print last validation loss to console for Bayesian Optimisation
    with open(args.debugging_file, 'a') as file:
        print("[main]: program ran successfully!", file=file)


model_map: Dict[str, type[ParentModel]] = {
    "U-Net": UNet,
    "P3D-ResNet": P3DResNet,
    "U-ResNet": UResNet,
    "FreMixer": FreMixer,
    "FracMixer": FracMixer,
}
if __name__ == '__main__':
    main()
