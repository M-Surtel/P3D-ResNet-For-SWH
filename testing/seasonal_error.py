import functools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import argparse
import pandas as pd
import polars as pl
import numpy as np
from typing import Dict
from enum import Enum, auto
import calendar
import re

from model_classes.abstract_model import ParentModel
from model_classes.persistence_model import PersistenceModel
from model_classes.P3D_ResNet import P3DResNet
from model_classes.unet import UNet
from training.train import parse_model_args
from training.train import get_dataset_files


class Season(Enum):
    WINTER = auto()
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()

    def rows(self, year):
        # defines the number of rows that are in a season (i.e. 24 * days in the season)
        match self:
            case Season.WINTER:
                if calendar.isleap(year):
                    return 2184
                else:
                    return 2160
            case Season.SPRING:
                return 2208
            case Season.SUMMER:
                return 2208
            case Season.AUTUMN:
                return 2184

    def next(self, year):
        match self:
            case Season.WINTER:
                return Season.SPRING, year + 1
            case Season.SPRING:
                return Season.SUMMER, year
            case Season.SUMMER:
                return Season.AUTUMN, year
            case Season.AUTUMN:
                return Season.WINTER, year


class SeasonTracker:
    def __init__(self, starting_year: int, starting_season: Season, starting_row: int):
        self.year = starting_year
        self.season = starting_season
        self.rows = starting_row
        self.rows_in_season = starting_season.rows(starting_year)

    def add_rows(self, num_rows):
        new_rows = self.rows + num_rows
        if new_rows < self.rows_in_season:
            self.rows += num_rows
            return {self.season: num_rows}
        elif new_rows >= self.rows_in_season:
            new_season_rows = new_rows - self.rows_in_season
            old_season_rows = num_rows - new_season_rows
            old_season = self.season
            self.season, self.year = self.season.next(self.year)
            self.rows = new_season_rows
            return {old_season: old_season_rows, self.season: new_season_rows}


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
    parser.add_argument("-p", "--point-location", type=str,
                        help="(Required) Centre point location given in tuple form (e.g. '(25, 30)', '(LAT, LON)').")
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

    test_data = get_dataset_files(args.data[0])  # load data
    error_dicts = []  # initialise list of dictionaries for storing loss and categorising it by season
    season_tracker = SeasonTracker(2020, Season.WINTER, 744)

    # spatial indices (3x3 surrounding the point)
    numbers = re.findall(r'\d+', args.point_location)
    lat_idx, lon_idx = map(int, numbers)
    spatial_idxs = (slice(lat_idx-1, lat_idx+2), slice(lon_idx-1, lon_idx+2))

    def batch(files, batch_size, preprocessing):
        def decorator(func):
            @functools.wraps(func)
            def wrapper():
                window_size = batch_size + model.lookback + model.lead_time  # The total raw data points needed to create one full batch
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
                        batch_x, batch_y = preprocessing(raw_window)
                        func(batch_x, batch_y)
                        buffer = buffer[batch_size:]  # slide buffer forward
                if len(buffer) > model.lookback + model.lead_time:  # enough data to create at least one more data point
                    batch_x, batch_y = preprocessing(buffer)
                    func(batch_x, batch_y)
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

        outputs = model.swh_reduce(outputs)
        outputs = outputs[:, *spatial_idxs]
        labels = model.swh_reduce(labels)
        labels = labels[:, *spatial_idxs]

        error: torch.Tensor = torch.abs(outputs - labels)  # absolute error
        pixels_per_sample = error[0].numel()
        batch_dict = {'season': [], 'mae': []}
        season_dict = season_tracker.add_rows(len(error))
        starting_row = 0
        for season, num_rows in season_dict.items():
            batch_dict['season'].extend([season.name] * num_rows * pixels_per_sample)
            chunk = error[starting_row: starting_row + num_rows]
            batch_dict['mae'].extend(chunk.flatten().tolist())
            starting_row += num_rows
        error_dicts.append(batch_dict)

    # determine output features we care about the error for
    error_output_indices = [args.output_features.index(error_output_feature) for error_output_feature in
                            args.error_output_features]

    calculate_errors()  # running of the stuff...
    # convert the list of batch_dicts into an efficient polars dataframe
    season_categories = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
    season_dtype = pl.Enum(season_categories)
    df = pl.concat([
        pl.DataFrame(batch) for batch in error_dicts
    ])
    df = df.with_columns(
        pl.col('season').cast(season_dtype),
        pl.col('mae').cast(pl.Float32)
    )

    # print errors to file
    with open(args.debugging_file, 'a') as file:
        print("[main]: creating boxplot", file=file)

    # create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.to_pandas(), x='season', y='mae', order=season_categories)

    plt.title('Absolute Error Distribution by Season')
    plt.xlabel('Season')
    plt.ylabel('Absolute Error (m)')
    save_path = Path(f"{args.save_path}-boxplot")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

    # save data for remaking plots
    df.write_parquet(f'{args.debugging_file}-polars-df.parquet')


if __name__ == '__main__':
    main()
