# P3D-ResNet-For-SWH
This repo provides accompanying code for a paper titled "Incorporating Physical Considerations into Deep Learning for Predicting the Significant Wave Height of Ocean Waves" that is currently yet to be published.

<!-- toc -->
- [How to Use This Repository](#how-to-use-this-repository)
- [Documentation](#documentation)
    - [Dataset Creation](#dataset-creation)
        - [ERA5DataAcquisition.py](#ERA5DataAcquisition)
        - [createERA5Dataset.py](#createERA5Dataset)
        - [calculate_feature_means.py](#calculate-feature-means)
        - [save_sea_mask.py](#save-sea-mask)
        - [append_var_to_npy_dataset.py](#append-var-to-npy-dataset)
        - [crop_npy_spatial.py](#crop-npy-spatial)
        - [crop_to_divisible.py](#crop-to-divisible)
        - [decrease_resolution.py](#decrease-resolution)
    - [Model Classes](#model-classes)
        - [P3D_ResNet_blocks](#P3D-ResNet-blocks)
            - [pytorch_blocks.py](#pytorch-blocks)
        - [abstract_model.py](#abstract-model)
        - [custom_loss.py](#custom-loss)
        - [P3D_ResNet.py](#P3D-ResNet)
        - [persistence_model.py](#persistence-model)
        - [shapley_P3D_ResNet.py](#shapley-P3D-ResNet)
        - [U_ResNet.py](#U-ResNet)
        - [unet.py](#unet)
    - [Shapley Values](#shapley-values)
        - [captum_shapley_sampling.py](#captum-shapley-sampling)
    - [Testing](#testing)
        - [total_error_statistics.py](#total-error-statistics)
        - [error_vs_feature.py](#error-vs-feature)
        - [error_vs_swh.py](#error-vs-swh)
        - [seasonal_error.py](#seasonal-error)
        - [test_pixel_change.py](#test-pixel-change)
    - [Training](#training)
        - [train.py](#train.py)
- [Model Configurations](#model-configurations)
<!-- tocstop -->

## How To Use This Repository
This repository exists so that the results of "Incorporating Physical Considerations into Deep Learning for Predicting the Significant Wave Height of Ocean Waves" can be replicated. The [Documentation](#documentation) section explains the purpose of each script, while the [Model Configurations](#model-configurations) section notes the exact configurations of the P3D-ResNet used in the paper.

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


## Model Configurations