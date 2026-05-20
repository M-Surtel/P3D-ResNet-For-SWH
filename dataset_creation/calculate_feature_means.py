import numpy as np
import torch
import pandas as pd
import argparse
from typing import Dict

from training.train import get_dataset_files


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--input-normalisation-file", type=str, required=True,
                             help="(Required) min-max input normalisation is applied using the min-max value provided in file."
                                  "Input normalisation file is needed for determining indices of input/output variables.")
    parser.add_argument("-d", "--data-files", type=str, required=True,
                                help="(Required) Path relative to this script to the dataset file."
                                     "\n If the filename conforms to the regular expression '*(0).npy', "
                                     "this script will automatically try to find files of '[Name](N).npy' "
                                     "until it reaches an N for which there is no file.")
    return parser.parse_args()


def calculate_means(data_files, df):
    feature_to_sum: Dict[str, float] = {feature: 0 for feature in df.feature}
    data_counter = 0
    for file in data_files:
        data = torch.from_numpy(np.load(file))
        data_counter += data[:, 0].numel()
        for i, feature in enumerate(feature_to_sum.keys()):
            feature_to_sum[feature] += float(torch.sum(data[:, i]))
    return feature_to_sum, data_counter


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.set_default_device(device)

    args = parse_user_args()
    df = pd.read_csv(args.input_normalisation_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    data_files = get_dataset_files(args.data_files)

    if len(data_files) == 0:
        raise ValueError('[ERROR] data_files is empty! --data-files option has been improperly specified!')

    feature_to_sum, data_counter = calculate_means(data_files, df)

    df['mean'] = [feature_sum / data_counter for feature_sum in feature_to_sum.values()]
    df.to_csv(args.input_normalisation_file.removesuffix('.csv') + '-mean.csv')


if __name__ == '__main__':
    main()
