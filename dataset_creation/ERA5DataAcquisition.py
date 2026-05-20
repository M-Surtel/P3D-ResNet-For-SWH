import cdsapi
import os
from pathlib import Path

# This script downloads ALL data of the given parameters
# Files that already exist are skipped
c = cdsapi.Client()
folder_name = './data/ERA5/japan-sst'
Path(folder_name).mkdir(exist_ok=True, parents=True)
file_name = 'japan-sst'
for year in range(2023, 1986, -1):
    for month_name, month_values in (('JanToApr', ['01', '02', '03', '04']), ('MayToAug', ['05', '06', '07', '08']),
                                     ('SepToDec', ['09', '10', '11', '12'])):
        file = f'{folder_name}/{year}-{month_name}-{file_name}.grib'
        if not os.path.exists(file):
            c.retrieve(
                name='reanalysis-era5-single-levels',
                request=
                {
                    'product_type': ["reanalysis"],
                    'area': [
                        62.5, 115, 15, 162.5
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'month': month_values,
                    'year': [year],
                    "variable": [
                        "sea_surface_temperature",
                        # "mean_wave_direction",
                        # "mean_wave_period",
                        # "significant_height_of_combined_wind_waves_and_swell",
                        # "air_density_over_the_oceans",
                        # "free_convective_velocity_over_the_oceans",
                        # "maximum_individual_wave_height",
                        # "mean_direction_of_total_swell",
                        # "mean_direction_of_wind_waves",
                        # "mean_period_of_total_swell",
                        # "mean_period_of_wind_waves",
                        # "mean_square_slope_of_waves",
                        # "mean_wave_period_based_on_second_moment_for_swell",
                        # "mean_wave_period_based_on_second_moment_for_wind_waves",
                        # "mean_zero_crossing_wave_period",
                        # "model_bathymetry",
                        # "ocean_surface_stress_equivalent_10m_neutral_wind_direction",
                        # "ocean_surface_stress_equivalent_10m_neutral_wind_speed",
                        # "peak_wave_period",
                        # "period_corresponding_to_maximum_individual_wave_height",
                        # "significant_height_of_total_swell",
                        # "significant_height_of_wind_waves",
                        # "wave_spectral_directional_width",
                        # "wave_spectral_directional_width_for_swell",
                        # "wave_spectral_directional_width_for_wind_waves",
                        # "wave_spectral_kurtosis",
                        # "wave_spectral_peakedness",
                        # "wave_spectral_skewness",
                        # "benjamin_feir_index",
                    ],
                    "data_format": "grib",
                    "download_format": "unarchived",
                },
            target=file)
