import torch
import torch.nn as nn
from model_classes.abstract_model import ParentModel


class PersistenceModel(ParentModel):
    def __init__(self, feature_list, input_features, output_features, lead_time=1, spatial_reduction=0, **kwargs):
        super().__init__()
        spatial_reduction //= 2  # padding of 1 reduces reduction by 2, so we half reductions to make the conversion easier
        self._lookback = 0
        self._lead_time = lead_time
        self._loss_fn = nn.MSELoss()
        self._reduce_swh = lambda swh_tensor: torch.sum(swh_tensor**2 / 16, 1)**0.5 * 4
        self._optimiser = None
        self._scheduler = None

        def _crop_output(tensor):
            match len(tensor.shape):
                case 4:
                    return tensor[:, :, spatial_reduction: -spatial_reduction, spatial_reduction: -spatial_reduction]
                case 2:
                    return tensor[spatial_reduction: -spatial_reduction, spatial_reduction: -spatial_reduction]
                case _:
                    raise ValueError("[_crop_output]: shape must either be of size 2 or 4")

        self._crop_output = _crop_output
        if spatial_reduction == 0:
            self._crop_output = lambda tensor: tensor
        self._spatial_reduction = spatial_reduction

        # determine input/output indices for use in preprocess_data()
        self.input_indices = [feature_list.index(feature) for feature in input_features]
        self.output_indices = [feature_list.index(feature) for feature in output_features]

    # We do a little bit of stealing from P3D_ResNet's preprocess_data()
    # Just removed lookback and further altered 1 line
    def preprocess_data(self, data: torch.Tensor):
        # input shape (T, F, Lat, Lon)
        # -> output1: x, shape (N, F, T, Lat, Lon)
        # -> output2: y, shape (N, F, T, Lat, Lon)
        def create_sequences(data):
            # creates temporal windows
            # input shape (F, T, Lat, Lon)
            # -> output1: x, shape (N, F, T, Lat, Lon)
            # -> output2: y, shape (N, F, Lat, Lon)
            x, y = [], []
            for i in range(data.shape[1] - self.lead_time):
                feature = data[self.input_indices, i]  # <-- only acc altered line
                # forecasting target is one time step after input data
                target = data[self.output_indices, i + self.lead_time]
                x.append(feature)
                y.append(target)
            return torch.stack(x), self.crop_output(torch.stack(y))

        data = torch.permute(data, [1, 0, 2, 3])
        return create_sequences(data)

    def forward(self, x):
        return self.crop_output(x)

    @property
    def lookback(self):
        return self._lookback

    @property
    def lead_time(self):
        return self._lead_time

    @property
    def spatial_reduction(self):
        return self._spatial_reduction

    @property
    def loss_function(self):
        return self._loss_fn

    @property
    def swh_reduce(self):
        return self._reduce_swh

    @property
    def optimiser(self):
        return self._optimiser

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def crop_output(self):
        return self._crop_output
