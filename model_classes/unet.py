import torch
import torch.nn as nn
import torch.optim as optim
from queue import LifoQueue
from .abstract_model import ParentModel
from .custom_loss import MSExMAPELoss


class _CroppingLayer(nn.Module):
    def __init__(self, next_kernel_sizes):
        super().__init__()
        # calculate how much shrinking will occur due to unpadded convolutions
        spatial = []
        temporal = []
        for i in range(len(next_kernel_sizes)):
            spatial_res = 2 ** (i + 1) * (next_kernel_sizes[i] - 1)
            if i + 1 == len(next_kernel_sizes):
                spatial_res *= 2  # 2 conv layers in final layer
                temporal_res = 2 * 2  # 2 layers of 3-wide conv
            else:
                spatial_res *= 4  # 4 conv layers in normal layers
                temporal_res = 4 * 2  # 4 layers of 3-wide conv
            spatial.append(spatial_res)
            temporal.append(temporal_res)
        self.spatial_shrinking = sum(spatial)
        self.temporal_shrinking = sum(temporal)

    def forward(self, x):
        # x of shape (N, F, T, Lat, Lon)
        return x[:, :, self.temporal_shrinking // 2: x.shape[2] - self.temporal_shrinking // 2:,
               self.spatial_shrinking // 2: x.shape[3] - self.spatial_shrinking // 2,
               self.spatial_shrinking // 2: x.shape[4] - self.spatial_shrinking // 2]


class _IdentityLayer(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x


class UNet(ParentModel):
    def __init__(self, feature_list, input_features, output_features, lookback, lead_time=1, step_size=None,
                 gamma=None, max_min_list=None, norm_type=None, num_groups=1, kernel_list=(3, 3, 3, 3), padding='valid',
                 optimiser='Adam', additional_output_features=None, loss='MSE', loss_weight = 1, **kwargs):
        def _create_norm_layer(features) -> nn.Module:
            if norm_type == 'batch_norm':
                norm_layer = nn.BatchNorm3d(features)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=features)
            elif norm_type is None:
                norm_layer = None
            else:
                raise ValueError(f"Invalid norm_type: {norm_type}")
            return norm_layer
        super().__init__()
        assert padding == 'valid' or padding == 'same'
        self.uses_valid_padding = True if padding == 'valid' else False

        self._crop_output = lambda tensor: tensor
        self.crop_input = False

        # layers initialisation
        self.layers = nn.ModuleDict()
        self.num_resolutions = len(kernel_list)
        num_features = len(input_features)
        # create left and right sides for each resolution
        for i in range(self.num_resolutions):
            input_dim_factor = 2 ** (i - 1) if i != 0 else 1
            output_dim_factor = 2 ** i

            # left side
            left_list = nn.ModuleList()
            if i != 0:
                left_list.append(nn.MaxPool3d((1, 2, 2), (1, 2, 2)))
            left_list.append(nn.Conv3d(input_dim_factor * num_features, output_dim_factor * num_features,
                                       (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            left_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features,
                                       (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            self.layers[f'left{i}'] = left_list

            # don't run right side for lowest resolution
            if i == self.num_resolutions - 1:
                left_list.append(
                    nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2),
                                       (1, 2, 2)))
                # left list is heap memory so self.layers will automatically be updated; no need for "self.layers[f'left{i}'] = left_list"
                continue

            # right side
            right_list = nn.ModuleList()
            right_list.append(nn.Conv3d(2 * output_dim_factor * num_features, output_dim_factor * num_features,
                                        (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            right_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features,
                                        (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            if i != 0:
                right_list.append(
                    nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2),
                                       (1, 2, 2)))
            self.layers[f'right{i}'] = right_list
        # create cropping layer for each layer except last layer
        if self.uses_valid_padding:
            self.cropping_layers = nn.ModuleList([_CroppingLayer(kernel_list[i+1:]) for i in range(len(kernel_list) - 1)])
        else:
            # insert identity layer if no cropping is required
            self.cropping_layers = nn.ModuleList([_IdentityLayer() for _ in range(len(kernel_list) - 1)])
        # calculate how much shrinking will occur due to unpadded convolutions before the linear final layer
        temporal = []
        for i in range(len(kernel_list)):
            if i + 1 == len(kernel_list):
                temporal_res = 2 * 2  # 2 layers of 3-wide conv
            else:
                temporal_res = 4 * 2  # 4 layers of 3-wide conv
            temporal.append(temporal_res)
        temporal_shrinking = sum(temporal) if self.uses_valid_padding else 0
        assert lookback + 1 - temporal_shrinking > 0
        self.final_layer = nn.Linear(num_features * (lookback + 1 - temporal_shrinking), 1)

        self._lookback = lookback
        self._lead_time = lead_time
        self._spatial_reduction = 0
        self._reduce_swh = lambda swh_tensor: torch.sum(swh_tensor ** 2 * 0.0625, 1) ** 0.5 * 4  # (* 0.0625) == (/ 16)
        # expects output of shape (N, F, Lat, Lon)
        match loss:
            case 'MSE':
                self._loss_fn = nn.MSELoss()
            case 'MSExMAPE':
                self._loss_fn = MSExMAPELoss(weight=loss_weight)

        # initialise optimiser and scheduler
        try:
            optimiser_class: torch.optim = eval(f'optim.{optimiser}')
        except AttributeError:
            raise ValueError(f'[ERROR]: incorrect optimiser; there is no torch.optim.{optimiser}')
        # only pass kwargs to optimiser that are in the optimiser's parameters
        optimiser_args = [var for var in optimiser_class.__init__.__code__.co_varnames if var != 'self']
        optimiser_kwargs = {key: value for key, value in kwargs.items() if key in optimiser_args}
        self._optimiser = optimiser_class(self.parameters(), **optimiser_kwargs)
        self._scheduler: optim.lr_scheduler.StepLR | None
        if step_size is None or gamma is None:
            self._scheduler = None
        else:
            self._scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=step_size, gamma=gamma)

        # input normalisation
        if max_min_list is not None:
            self.maxima = [max_min_tuple[0] for max_min_tuple in max_min_list]
            self.minima = [max_min_tuple[1] for max_min_tuple in max_min_list]
        self.input_norm = max_min_list is not None

        # determine input/output indices for use in preprocess_data()
        self.input_indices = [feature_list.index(feature) for feature in input_features]
        self.output_indices = [feature_list.index(feature) for feature in output_features]
        if additional_output_features is not None:
            self.additional_indices = [feature_list.index(feature) for feature in additional_output_features]
            self.additional_output = True
        else:
            self.additional_output = False

        # calculate output shrinking due to unpadded convolutions
        spatial = []
        for i in range(len(kernel_list)):
            spatial_res = 2 ** i * (kernel_list[i] - 1)
            if i + 1 == len(kernel_list):
                spatial_res *= 2  # 2 conv layers in final layer
            else:
                spatial_res *= 4  # 4 conv layers in normal layers
            spatial.append(spatial_res)
        self._output_shrinking = sum(spatial) if self.uses_valid_padding else 0


    @property
    def lookback(self):
        return self._lookback

    @property
    def lead_time(self):
        return self._lead_time

    @property
    def loss_function(self):
        return self._loss_fn

    @property
    def spatial_reduction(self):
        return self._spatial_reduction

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

    def forward(self, x):
        # x shape is (N, F, T, Lat, Lon)
        if self.input_norm:
            for feature in range(x.shape[1]):
                maxima_tensor = torch.as_tensor(self.maxima).repeat((x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
                minima_tensor = torch.as_tensor(self.minima).repeat((x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
                x = (x - minima_tensor) / (maxima_tensor - minima_tensor)

        vertical_output = x
        horizontal_outputs = LifoQueue(maxsize=self.num_resolutions - 1)

        # left side
        for i in range(self.num_resolutions):
            module_list = self.layers[f'left{i}']
            for module in module_list:
                vertical_output = module(vertical_output)
            if i != self.num_resolutions - 1:  # last U-Net layer has no right side
                horizontal_outputs.put(self.cropping_layers[i](vertical_output))

        # right side
        # iterate from self.num_resolutions back to 0 (inclusive)
        for i in range(self.num_resolutions - 2, -1, -1):
            module_list = self.layers[f'right{i}']
            vertical_output = torch.concat([vertical_output, horizontal_outputs.get()], dim=1)
            for module in module_list:
                vertical_output = module(vertical_output)

        # go from shape (N, F, T, Lat, Lon) -> (N, F * T, Lat, Lon)
        out = torch.reshape(vertical_output, (vertical_output.shape[0], vertical_output.shape[1] * vertical_output.shape[2], vertical_output.shape[3], vertical_output.shape[4]))
        out = out.permute((0, 2, 3, 1))
        out = self.final_layer(out)
        out = torch.squeeze(out)
        out = torch.unsqueeze(out, 1)

        if self.input_norm:
            minima = torch.full_like(out, self.minima[-1])
            maxima = torch.full_like(out, self.maxima[-1])
            out = out * (maxima - minima) + minima
        return out

    def preprocess_data(self, data: torch.Tensor):
        # input shape (T, F, Lat, Lon)
        # -> output1: x, shape (N, F, T, Lat, Lon)
        # -> output2: y, shape (N, F, T, Lat, Lon)
        # -> (opt.) output3: z, shape (N, F, T, Lat, Lon)
        def create_sequences(data, lookback):
            # creates temporal windows
            # input shape (F, T, Lat, Lon)
            # -> output1: x, shape (N, F, T, Lat, Lon)
            # -> output2: y, shape (N, F, Lat, Lon)
            # -> (opt.) output3: z, shape (N, F, Lat, Lon)
            x, y, z = [], [], []
            for i in range(data.shape[1] - lookback - self.lead_time):
                feature = data[self.input_indices, i:i + lookback + 1]
                # forecasting target is self.lead_time time step(s) after input data
                target = data[self.output_indices, i + lookback + self.lead_time]
                x.append(feature)
                y.append(target)

                if self.additional_output:
                    additional_output = data[self.additional_indices, i + lookback + self.lead_time]
                    z.append(additional_output)

            if self.crop_input:
                x = self.crop_output(torch.stack(x))
            else:
                x = torch.stack(x)

            if self.additional_output:
                return x, self.crop_output(torch.stack(y)), self.crop_output(torch.stack(z))
            return x, self.crop_output(torch.stack(y))

        with torch.no_grad():
            data = torch.permute(data, [1, 0, 2, 3])
            return create_sequences(data, self.lookback)


class OldUNet(ParentModel):
    def __init__(self, num_features, lookback, lr=0.001, weight_decay=0.005, step_size=None, gamma=None, max_min_list=None, norm_type=None, num_groups=1, kernel_list=(3, 3, 3, 3), padding='valid'):
        def _create_norm_layer(features) -> nn.Module:
            if norm_type == 'batch_norm':
                norm_layer = nn.BatchNorm3d(features)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=features)
            elif norm_type is None:
                norm_layer = None
            else:
                raise ValueError(f"Invalid norm_type: {norm_type}")
            return norm_layer
        super().__init__()
        assert padding == 'valid' or padding == 'same'
        self.uses_valid_padding = True if padding == 'valid' else False

        # layers initialisation
        self.layers = nn.ModuleDict()
        self.num_resolutions = len(kernel_list)
        # create left and right sides for each resolution
        for i in range(self.num_resolutions):
            input_dim_factor = 2 ** (i - 1) if i != 0 else 1
            output_dim_factor = 2 ** i

            # left side
            left_list = nn.ModuleList()
            if i != 0:
                left_list.append(nn.MaxPool3d((1, 2, 2), (1, 2, 2)))
            left_list.append(nn.Conv3d(input_dim_factor * num_features, output_dim_factor * num_features,
                                       (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            left_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features,
                                       (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            self.layers[f'left{i}'] = left_list

            # don't run right side for lowest resolution
            if i == self.num_resolutions - 1:
                left_list.append(
                    nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2),
                                       (1, 2, 2)))
                # left list is heap memory so self.layers will automatically be updated; no need for "self.layers[f'left{i}'] = left_list"
                continue

            # right side
            right_list = nn.ModuleList()
            right_list.append(nn.Conv3d(2 * output_dim_factor * num_features, output_dim_factor * num_features,
                                        (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            right_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features,
                                        (3, kernel_list[i], kernel_list[i]), padding=padding))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            if i != 0:
                right_list.append(
                    nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2),
                                       (1, 2, 2)))
            self.layers[f'right{i}'] = right_list
        # create cropping layer for each layer except last layer
        if self.uses_valid_padding:
            self.cropping_layers = nn.ModuleList([_CroppingLayer(kernel_list[i+1:]) for i in range(len(kernel_list) - 1)])
        else:
            # insert identity layer if no cropping is required
            self.cropping_layers = nn.ModuleList([_IdentityLayer() for _ in range(len(kernel_list) - 1)])
        # calculate how much shrinking will occur due to unpadded convolutions before the linear final layer
        temporal = []
        for i in range(len(kernel_list)):
            if i + 1 == len(kernel_list):
                temporal_res = 2 * 2  # 2 layers of 3-wide conv
            else:
                temporal_res = 4 * 2  # 4 layers of 3-wide conv
            temporal.append(temporal_res)
        temporal_shrinking = sum(temporal) if self.uses_valid_padding else 0
        self.final_layer = nn.Linear(num_features * (lookback + 1 - temporal_shrinking), 1)

        self._lookback = lookback
        self._loss_fn = nn.MSELoss()

        # initialise optimiser and scheduler
        self._optimiser = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if step_size is None or gamma is None:
            self._scheduler = None
        else:
            self._scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=step_size, gamma=gamma)

        # input normalisation
        if max_min_list is not None:
            self.maxima = [max_min_tuple[0] for max_min_tuple in max_min_list]
            self.minima = [max_min_tuple[1] for max_min_tuple in max_min_list]
        self.input_norm = max_min_list is not None

        # calculate output shrinking due to unpadded convolutions
        spatial = []
        for i in range(len(kernel_list)):
            spatial_res = 2 ** i * (kernel_list[i] - 1)
            if i + 1 == len(kernel_list):
                spatial_res *= 2  # 2 conv layers in final layer
            else:
                spatial_res *= 4  # 4 conv layers in normal layers
            spatial.append(spatial_res)
        self._output_shrinking = sum(spatial) if self.uses_valid_padding else 0


    @property
    def lookback(self):
        return self._lookback

    @property
    def loss_function(self):
        return self._loss_fn

    @property
    def optimiser(self):
        return self._optimiser

    @property
    def scheduler(self):
        return self._scheduler

    def forward(self, x):
        # x shape is (N, F, T, Lat, Lon)
        if self.input_norm:
            for feature in range(x.shape[1]):
                minima = torch.full_like(x[:, feature], self.minima[feature])
                maxima = torch.full_like(x[:, feature], self.maxima[feature])
                x[:, feature] = (x[:, feature] - minima) / (maxima - minima)

        vertical_output = x
        horizontal_outputs = LifoQueue(maxsize=self.num_resolutions - 1)

        # left side
        for i in range(self.num_resolutions):
            module_list = self.layers[f'left{i}']
            for module in module_list:
                vertical_output = module(vertical_output)
            if i != self.num_resolutions - 1:  # last U-Net layer has no right side
                horizontal_outputs.put(self.cropping_layers[i](vertical_output))

        # right side
        # iterate from self.num_resolutions back to 0 (inclusive)
        for i in range(self.num_resolutions - 2, -1, -1):
            module_list = self.layers[f'right{i}']
            vertical_output = torch.concat([vertical_output, horizontal_outputs.get()], dim=1)
            for module in module_list:
                vertical_output = module(vertical_output)

        # go from shape (N, F, T, Lat, Lon) -> (N, F * T, Lat, Lon)
        out = torch.reshape(vertical_output, (vertical_output.shape[0], vertical_output.shape[1] * vertical_output.shape[2], vertical_output.shape[3], vertical_output.shape[4]))
        out = out.permute((0, 2, 3, 1))
        out = self.final_layer(out)
        out = torch.squeeze(out)

        if self.input_norm:
            minima = torch.full_like(out, self.minima[-1])
            maxima = torch.full_like(out, self.maxima[-1])
            out = out * (maxima - minima) + minima
        return out

    def preprocess_data(self, data):
        # input shape (T, F, Lat, Lon)
        # -> output1: x, shape (N, F, T, Lat, Lon)
        # -> output2: y, shape (N, F, T, Lat, Lon)
        def create_sequences(data, lookback):
            # creates temporal windows
            # input shape (F, T, Lat, Lon)
            # -> output1: x, shape (N, F, T, Lat, Lon)
            # -> output2: y, shape (N, F, T, Lat, Lon)
            x, y = [], []
            for i in range(data.shape[1] - lookback - 1):
                feature = data[:, i:i + lookback + 1, :, :]
                # forecasting target is one time step after input data
                target = data[-1, i + lookback + 1, :, :]
                x.append(feature)
                y.append(target)
            x = torch.stack(x)
            y = torch.stack(y)
            # account for potential shrinking due to unpadded convs (self._output_shrinking == 0 if same padding is used)
            y = y[:, self._output_shrinking // 2:y.shape[1] - self._output_shrinking // 2, self._output_shrinking // 2:y.shape[2] - self._output_shrinking // 2]
            return x, y

        data = torch.permute(data, [1, 0, 2, 3])
        return create_sequences(data, self.lookback)
