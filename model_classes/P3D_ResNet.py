import torch
import torch.nn as nn
import torch.optim as optim

from .abstract_model import ParentModel
from .P3D_ResNet_blocks.pytorch_blocks import P3DA, P3DB, P3DC
from .custom_loss import MSExMAPELoss


class P3DResNet(ParentModel):
    def __init__(self, feature_list, input_features, output_features, lookback, num_blocks, hidden_size=None,
                 lead_time=1,
                 norm_type=None, norm_before_activation=True, num_groups=1, dropout_rate=0.0, weight_norm=False,
                 optimiser='Adam', loss='MSE', loss_weight=1, step_size=None, gamma=None, max_min_list=None,
                 kernel_size=3,
                 spatial_reduction=0, spatial_reduction_strategy=None, padding_mode='zeros', num_final_layers=1,
                 additional_output_features=None, **kwargs):
        super().__init__()
        # determine & calculate how spatial reductions are handled
        self.crop_input = False
        padding_list: list[int | str]  # supports 'valid', 'same', and numerical padding
        spatial_reduction //= 2  # padding of 1 reduces reduction by 2, so we half reductions to make the conversion easier
        match spatial_reduction_strategy:
            case 'gradual':
                reductions_per_layer = (kernel_size - 1) // 2

                # divide reductions over layers
                num_layers = num_blocks * 3
                base_reductions = spatial_reduction // num_layers
                residual_reductions = spatial_reduction % num_layers
                reductions_list = [base_reductions] * num_layers
                for i in range(1, residual_reductions + 1):
                    reductions_list[-i] += 1
                # determine padding from reductions_list
                padding_list = [reductions_per_layer - reduction for reduction in reductions_list]
                for padding in padding_list:
                    assert padding >= 0
            case 'end':
                num_layers = num_blocks * 3
                padding_list = ['same'] * num_layers

                reductions_per_layer = (kernel_size - 1) // 2
                padding = reductions_per_layer - spatial_reduction
                assert padding >= 0
                padding_list[-1] = padding
            case _:
                if spatial_reduction != 0:
                    self.crop_input = True
                num_layers = num_blocks * 3
                padding_list = ['same'] * num_layers

        def _crop_output(tensor):
            # assumes the last two dimensions are Lat & Lon
            return tensor[..., spatial_reduction: -spatial_reduction, spatial_reduction: -spatial_reduction]

        self._crop_output = _crop_output
        if spatial_reduction == 0:
            self._crop_output = lambda tensor: tensor
        self._spatial_reduction = spatial_reduction

        # initialise model architecture
        num_features = len(input_features)
        if hidden_size is None:
            hidden_size = num_features
            self.initial_layer = lambda x: x
        else:
            self.initial_layer = nn.Conv3d(num_features, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.extend(
                [
                    P3DA(hidden_size, norm_type, num_groups, dropout_rate, weight_norm, kernel_size,
                         padding_list[i * 3], padding_mode, norm_before_activation),
                    P3DB(hidden_size, norm_type, num_groups, dropout_rate, weight_norm, kernel_size,
                         padding_list[i * 3 + 1], padding_mode, norm_before_activation),
                    P3DC(hidden_size, norm_type, num_groups, dropout_rate, weight_norm, kernel_size,
                         padding_list[i * 3 + 2], padding_mode, norm_before_activation)
                ])
        self.final_layers = nn.ModuleList()
        for i in range(num_final_layers - 1):
            self.final_layers.extend(
                [nn.Linear(hidden_size * (lookback + 1), hidden_size * (lookback + 1)), nn.ReLU()])
        self.final_layers.append(nn.Linear(hidden_size * (lookback + 1), len(output_features)))

        self._lookback = lookback
        self._lead_time = lead_time
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
        # print('############## New Forward ##############')
        # x shape is (N, F, T, Lat, Lon)
        if self.input_norm:
            # for feature in range(x.shape[1]):
            #     minima = torch.full_like(x[:, feature], self.minima[feature])
            #     maxima = torch.full_like(x[:, feature], self.maxima[feature])
            #     x[:, feature] = (x[:, feature] - minima) / (maxima - minima)
            maxima_tensor = torch.as_tensor(self.maxima).repeat(
                (x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
            minima_tensor = torch.as_tensor(self.minima).repeat(
                (x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
            x = (x - minima_tensor) / (maxima_tensor - minima_tensor)

        x = self.initial_layer(x)

        for block in self.blocks:
            x = block(x)

        # final layer to squish feature and time dimensions
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = x.permute((0, 2, 3, 1))
        for layer in self.final_layers:
            x = layer(x)

        x = x.permute(0, 3, 1, 2)

        if self.input_norm:
            minima = torch.full_like(x, self.minima[-1])
            maxima = torch.full_like(x, self.maxima[-1])
            x = x * (maxima - minima) + minima
        return x

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
