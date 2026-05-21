# P3D-ResNet-For-SWH
This repo provides accompanying code for a paper titled "Incorporating Physical Considerations into Deep Learning for Predicting the Significant Wave Height of Ocean Waves" that is currently yet to be published.

<!-- toc -->
- [How to Use This Repository](#how-to-use-this-repository)
- [Documentation](#documentation)
    - [Dataset Creation](#dataset-creation)
        - [ERA5DataAcquisition.py](#era5dataacquisitionpy)
        - [createERA5Dataset.py](#createera5datasetpy)
        - [calculate_feature_means.py](#calculatefeaturemeanspy)
        - [save_sea_mask.py](#saveseamaskpy)
        - [append_var_to_npy_dataset.py](#appendvartonpydatasetpy)
        - [crop_npy_spatial.py](#cropnpyspatialpy)
        - [crop_to_divisible.py](#croptodivisiblepy)
        - [decrease_resolution.py](#decreaseresolutionpy)
    - [Model Classes](#model-classes)
        - [P3D_ResNet_blocks](#P3DResNetblocks)
            - [pytorch_blocks.py](#pytorchblockspy)
        - [abstract_model.py](#abstractmodelpy)
        - [custom_loss.py](#customlosspy)
        - [P3D_ResNet.py](#P3DResNetpy)
        - [persistence_model.py](#persistencemodelpy)
        - [shapley_P3D_ResNet.py](#shapleyP3DResNetpy)
        - [U_ResNet.py](#UResNetpy)
        - [unet.py](#unetpy)
    - [Shapley Values](#shapley-values)
        - [captum_shapley_sampling.py](#captumshapleysamplingpy)
    - [Testing](#testing)
        - [total_error_statistics.py](#totalerrorstatisticspy)
        - [error_vs_feature.py](#errorvsfeaturepy)
        - [seasonal_error.py](#seasonalerrorpy)
        - [test_pixel_change.py](#testpixelchangepy)
    - [Training](#training)
        - [train.py](#trainpy)
- [Model Configurations](#model-configurations)
<!-- tocstop -->

## How To Use This Repository
This repository exists so that the results of 
"Incorporating Physical Considerations into Deep Learning for Predicting the Significant Wave Height of Ocean Waves" can be replicated. 
The [Documentation](#documentation) section explains the purpose of each script, 
while the [Model Configurations](#model-configurations) section notes the exact configurations of the baseline and final P3D-ResNet used in the paper.

## Documentation
This section briefly explains the use of each script. For more specific help about the command-line arguments, you can run
```bash
$ python -m [module_name].[script_name] --help
```
in the terminal. Note that you should **not** include the .py at the end of the script. For example:
```bash
$ python -m testing.total_error_statistics --help
```
All Python files have --help commands to explain their arguments, and they help explain what the program needs to function.

### Dataset Creation
#### ERA5DataAcquisition.py
Script to download the ERA5 .grib files. 
For more information on this visit: https://cds.climate.copernicus.eu/how-to-api. 
The values are hard-coded (i.e. no command-line args). 
Each year is split up into 3 files because CDS imposes limits on file sizes that can be queried.
This script skips files that have already been downloaded.

#### createERA5Dataset.py
This script takes .grib --input-files, and creates .npy files out of the selected --features.
The training and testing scripts only take .npy files as they are easier to work with.
The min-max normalisation values are also in this file under `feature_to_max_min`.
Directional encoding is set using --directional_encoding.
Also generates a .csv max-min normalisation file.

#### calculate_feature_means.py
Uses --data-files to calculate feature means and appends them to --input-normalisation-file.
Feature means are used for Shapley analysis.
If you do not do Shapley analysis, you will likely not need this.

#### save_sea_mask.py
Uses --dataset-file and --input-normalisation-file to ascertain where the sea is and where the land is.
Any grid cell that has at least one non-zero value for significant wave height is deemed to be sea.
The remaining values are considered land.
Some scripts require a sea-mask to be able to tell land from sea.

#### append_var_to_npy_dataset.py
Convenience script that appends features from .grib files to existing .npy datasets.
This script is handy for when you want to test new variables that were not in your original dataset.
It takes the .grib --input-files and desired --features, and appends them to an existing .npy dataset --output-file.

#### crop_npy_spatial.py
This script allows the user to crop a dataset to receive a spatial subset.

#### crop_to_divisible.py
This script was written to make the output area neatly divisible for the UNet.
If you do not use the UNet code, you will likely not need it.

#### decrease_resolution.py
Script that allows the given .npy dataset --input-files to be reduced in resolution.
The paper in question uses this to half the spatial dimensions for computational speed.
The --help option explains the syntax required for the slicing.


### Model Classes
#### P3D_ResNet_blocks
##### pytorch_blocks.py
This contains the code that is used by P3D-ResNet scripts in the parent folder to construct P3D-ResNets.

#### abstract_model.py
An abstract parent class for all models.
This ensures that all model classes have, at least, all the basic sets of building blocks, 
allowing them to be used in place of one another in the training and testing scripts.
If you want to write a new model within this repo, it would inherit from the `ParentModel`.

#### custom_loss.py
This script is used to create a hybrid loss.
The hybrid is a weighted combination of Mean Squared Error and Mean Absolute Percentage Error.
The results using this hybrid loss did not make it in the paper.

#### P3D_ResNet.py
This combines the P3D-blocks ([pytorch_blocks.py](#pytorchblockspy)) into a single model. 
It is used by the training and testing scripts.

#### persistence_model.py
The persistence model offers a baseline to compare any model performance against. 
The persistence model simply predicts that the significant wave height will not change.

##### shapley_P3D_ResNet.py
This script takes the P3D-ResNet, but modifies the `forward()` function to accommodate missing features,
as it is required for Shapley analysis.

##### U_ResNet.py
An alternative architecture that we briefly experimented with.
It has the varying resolutions of a UNet, and the modular blocks of a ResNet.
This architecture has been used before in the ML literature.

#### unet.py
We considered the UNet as an alternative strategy when it became apparent the P3D-ResNet would require 
increased spatial kernel size in the convolutional layers.
In our experience, for the SWH problem, this architecture is cheaper, but performs worse.


### Shapley Values
#### captum_shapley_sampling.py
This script uses [Captum](https://captum.ai/) to calculate Shapley values.
It includes many options and ranges from off-manifold magnitude-based Shapley values to on-manifold accuracy-based Shapley values.
Asymmetric Shapley Values are also an option, allowing the weighing of time steps via --time-step-asv or doing regular ASVs via
--ancestor-features and --descendant-features. A lot of the arguments are optional.

### Testing
#### total_error_statistics.py
This is the most general testing script.
It takes a model checkpoint --checkpoint and testing dataset --data as input,
and uses that to calculate MAE, MAPE, MSE, RMSE, and generate a histogram of errors and a spatial heatmap of MAE.
If your test dataset is large, it is recommended to lower --histogram-fraction from 1.0 to something like 0.1 or 0.05.
This prevents the creation of exceedingly large histograms.

#### error_vs_feature.py
This script is useful for evaluating how the model's error changes with differing values of a specific feature.
In the paper we use it to calculate the error versus the significant wave height.
The output is a .pt file that contains a dictionary.
The `error` key in the dictionary contains the tensor of errors, while the `[feature_name]` key contains the tensor of feature values.
This can be used to created plots and derive statistics.

#### seasonal_error.py
A script that categorises error by season, for seasonal analysis.
It outputs a polars .parquet file that contains a dataframe with MAE values and seasons.

#### test_pixel_change.py
Given specific point indices --pixels, the script calculates the predicted difference (`predicted_swh - input_swh`) 
and actual difference (`target_swh - input_swh`). 
The most recent significant wave height is taken as the `input_swh`.
In the paper, this script is used to evaluate how well the model captures the dynamics of a high-error, a medium-error, and a low-error point.

### Training
#### train.py
This script handles the training for all models.
The --model-args are fed straight to the chosen --model.
To know what the possible --model-args are, you should check the script for the model you are using.
It is important to note down what your --model-args are, as they must be supplied whenever you want to recreate the model
(e.g. during further training or testing).
After each epoch, the model saves a checkpoint, so progress is not lost when the script is interrupted.
The checkpoint consists of a dictionary that contains
```python
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
```
Sensible file-naming policies are recommended when training multiple models.
If the filename conforms to the regular expression `*(0).npy`,
this script will automatically try to find files of `[Name](N).npy`
until it reaches an N for which there is no file.
Datasets created by [createERA5Dataset.py](#createera5datasetpy) conform to this regular expression.


## Model Configurations

Baseline model:
```bash
$ export name='training/japan-years-30-features-8-lead-time-12-kernel-size-3-no-reduction-4'
$ python -m training.train \
    -m P3D-ResNet \
    -a lead_time=12 lr=0.001 optimiser=Adam lookback=11 num_blocks=3 norm_type='group_norm' \
       num_groups=4 dropout_rate=0 weight_decay=0.001 kernel_size=3 spatial_reduction=8 \
       padding_mode=zeros \
    -t "./data/datasets/ERA5/japan-32-features-training-half-res/1987To2016-japan-32-features-training(0).npy" \
    -v "./data/datasets/ERA5/japan-32-features-validation-half-res/2017To2019-japan-32-features-validation(0).npy" \
    -e 100 \
    -b 128 \
    -d './results/'$name \
    -o './model_parameters/'$name \
    -n "./data/datasets/ERA5/japan-32-features-training-half-res/1987To2016-japan-32-features-training-feature-max-min.csv" \
    -p float32 \
    -i mwp wind wmb dwi_sin dwi_cos mwd_sin mwd_cos swh \
    -u swh
```

Final model:
```bash
$ export name='training/japan-years-30-features-8-lead-time-12-no-dir-vars-width-8'
$ python -m training.train \
    -m P3D-ResNet \
    -a lead_time=12 lr=0.001 optimiser=Adam lookback=11 num_blocks=3 norm_type='group_norm' \
       num_groups=4 dropout_rate=0 weight_decay=0.001 kernel_size=9 spatial_reduction=8 \
       spatial_reduction_strategy='gradual' padding_mode=zeros hidden_size=8 \
    -t "./data/datasets/ERA5/japan-32-features-training-half-res/1987To2016-japan-32-features-training(0).npy" \
    -v "./data/datasets/ERA5/japan-32-features-validation-half-res/2017To2019-japan-32-features-validation(0).npy" \
    -e 100 \
    -b 128 \
    -d './results/'$name \
    -o './model_parameters/'$name \
    -n "./data/datasets/ERA5/japan-32-features-training-half-res/1987To2016-japan-32-features-training-feature-max-min.csv" \
    -p float32 \
    -i mwp wind wmb swh \
    -u swh
```