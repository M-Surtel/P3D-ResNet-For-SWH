import torch
import torch.nn as nn
import torch.optim as optim
from queue import LifoQueue
from .abstract_model import ParentModel


class UNetBlock(nn.Module):
    def __init__(self, num_features, norm_type=None, num_groups=1, kernel_list=(3, 3, 3, 3)):
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
            left_list.append(nn.Conv3d(input_dim_factor * num_features, output_dim_factor * num_features, (kernel_list[i], kernel_list[i], 3), padding='same'))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            left_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features, (kernel_list[i], kernel_list[i], 3), padding='same'))
            if norm_type is not None:
                left_list.append(_create_norm_layer(output_dim_factor * num_features))
            left_list.append(nn.ReLU())
            self.layers[f'left{i}'] = left_list

            # don't run right side for lowest resolution
            if i == self.num_resolutions - 1:
                left_list.append(nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2), (1, 2, 2)))
                # left list is heap memory so self.layers will automatically be updated; no need for "self.layers[f'left{i}'] = left_list"
                continue

            # right side
            right_list = nn.ModuleList()
            right_list.append(nn.Conv3d(2 * output_dim_factor * num_features, output_dim_factor * num_features, (kernel_list[i], kernel_list[i], 3), padding='same'))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            right_list.append(nn.Conv3d(output_dim_factor * num_features, output_dim_factor * num_features, (kernel_list[i], kernel_list[i], 3), padding='same'))
            if norm_type is not None:
                right_list.append(_create_norm_layer(output_dim_factor * num_features))
            right_list.append(nn.ReLU())
            if i != 0:
                right_list.append(nn.ConvTranspose3d(output_dim_factor * num_features, input_dim_factor * num_features, (1, 2, 2), (1, 2, 2)))
            self.layers[f'right{i}'] = right_list

    def forward(self, vertical_output):
        horizontal_outputs = LifoQueue(maxsize=self.num_resolutions - 1)

        # left side
        for i in range(self.num_resolutions):
            module_list = self.layers[f'left{i}']
            for module in module_list:
                vertical_output = module(vertical_output)
            if i != self.num_resolutions - 1:
                horizontal_outputs.put(vertical_output)

        # right side
        # iterate from self.num_resolutions back to 0 (inclusive)
        for i in range(self.num_resolutions - 2, -1, -1):
            module_list = self.layers[f'right{i}']
            vertical_output = torch.concat([vertical_output, horizontal_outputs.get()], dim=1)
            for module in module_list:
                vertical_output = module(vertical_output)
        return vertical_output


class UResNet(ParentModel):
    def __init__(self, num_features, lookback, num_blocks, norm_type=None, num_groups=1,
                 lr=0.001, weight_decay=0.005, step_size=None, gamma=None, max_min_list=None, kernel_list=(3, 3, 3, 3)):
        super().__init__()
        # initialise model architecture
        self.blocks = nn.ModuleList([UNetBlock(num_features, norm_type, num_groups, kernel_list) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(num_features * (lookback + 1), 1)

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

        for block in self.blocks:
            x = block(x)

        # final layer to squish feature and time dimensions
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = x.permute((0, 2, 3, 1))
        x = self.final_layer(x)
        x = torch.squeeze(x)

        if self.input_norm:
            minima = torch.full_like(x, self.minima[-1])
            maxima = torch.full_like(x, self.maxima[-1])
            x = x * (maxima - minima) + minima
        return x

    def preprocess_data(self, data: torch.Tensor):
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
            return torch.stack(x), torch.stack(y)

        data = torch.permute(data, [1, 0, 2, 3])
        return create_sequences(data, self.lookback)
