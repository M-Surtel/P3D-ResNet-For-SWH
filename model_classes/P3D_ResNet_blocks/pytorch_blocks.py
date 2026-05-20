import torch
import torch.nn as nn
from abc import abstractmethod

class _MeanOnlyBatchNorm3d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(_MeanOnlyBatchNorm3d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3, 4])  # Calculate mean across N, D, H, W dimensions

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()

            x = x - batch_mean.view(1, -1, 1, 1, 1)  # Subtract mean, keeping dimensions
        else:
            x = x - self.running_mean.view(1, -1, 1, 1, 1)

        x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)

        return x


class P3DBlock(nn.Module):
    def __init__(self, num_features: int, norm_type: str, num_groups: int, dropout_rate: float, weight_norm: bool, kernel_size: int, spatial_padding: str | int, padding_mode: str, norm_before_activation: bool):
        def _create_norm_layer() -> nn.Module:
            match norm_type:
                case 'batch_norm':
                    norm_layer = nn.BatchNorm3d(num_features)
                case 'group_norm':
                    norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
                case 'mean_only_batch_norm':
                    norm_layer = _MeanOnlyBatchNorm3d(num_features)
                case None:
                    norm_layer = None
                case _:
                    raise ValueError(f"Invalid norm_type: {norm_type}")
            return norm_layer

        def _create_norm_activation_seq(activation_layer, norm_layer, norm_before):
            if norm_layer is None:
                return activation_layer
            if norm_before:
                return nn.Sequential(norm_layer, activation_layer)
            return nn.Sequential(activation_layer, norm_layer)

        super().__init__()

        match spatial_padding:
            case 'same':
                spatial_padding = (kernel_size - 1) // 2
            case 'valid':
                spatial_padding = 0
            case _:
                assert type(spatial_padding) == int
        temporal_padding = (kernel_size - 1) // 2  # i.e. same padding

        self.spatial_reduction = (kernel_size - 1) - spatial_padding * 2
        self.crop_skip_connection = lambda tensor: tensor[:, :, :, self.spatial_reduction // 2: -self.spatial_reduction // 2, self.spatial_reduction // 2: -self.spatial_reduction // 2]
        if self.spatial_reduction == 0:
            self.crop_skip_connection = lambda tensor: tensor

        self.first_layer = nn.Conv3d(num_features, num_features, (1, 1, 1), padding=(0, 0, 0))
        self.spatial_conv = nn.Conv3d(num_features, num_features, (1, kernel_size, kernel_size), padding=(0, spatial_padding, spatial_padding), padding_mode=padding_mode)
        self.temporal_conv = nn.Conv3d(num_features, num_features, (kernel_size, 1, 1), padding=(temporal_padding, 0, 0))
        self.last_layer = nn.Conv3d(num_features, num_features, (1, 1, 1), padding=(0, 0, 0))
        if weight_norm:
            self.first_layer = nn.utils.parametrizations.weight_norm(self.first_layer)
            self.spatial_conv = nn.utils.parametrizations.weight_norm(self.spatial_conv)
            self.temporal_conv = nn.utils.parametrizations.weight_norm(self.temporal_conv)
            self.last_layer = nn.utils.parametrizations.weight_norm(self.last_layer)

        # Combine normalisation and activations to avoid control flow statements in forward()
        self.relu1 = _create_norm_activation_seq(nn.ReLU(), _create_norm_layer(), norm_before_activation)
        self.relu2 = _create_norm_activation_seq(nn.ReLU(), _create_norm_layer(), norm_before_activation)
        self.relu3 = _create_norm_activation_seq(nn.ReLU(), _create_norm_layer(), norm_before_activation)
        self.relu4 = _create_norm_activation_seq(nn.ReLU(), _create_norm_layer(), norm_before_activation)
        self.dropout = nn.Dropout3d(dropout_rate)

    @abstractmethod
    def forward(self, x: torch.Tensor):
        return x


class P3DA(P3DBlock):
    def forward(self, x: torch.Tensor):
        # x shape (N, F, T, Lat, Lon)
        out = self.first_layer(x)
        out = self.relu1(out)

        out = self.spatial_conv(out)
        out = self.relu2(out)

        out = self.temporal_conv(out)
        out = self.relu3(out)

        out = self.last_layer(out)

        out = self.dropout(out)
        out = self.relu4(out + self.crop_skip_connection(x))

        return out


class P3DB(P3DBlock):
    def forward(self, x: torch.Tensor):
        # x shape (N, F, T, Lat, Lon)
        out = self.first_layer(x)
        out = self.relu1(out)

        spatial_out = self.spatial_conv(out)
        spatial_out = self.relu2(spatial_out)

        temporal_out = self.temporal_conv(out)
        out = self.relu3(self.crop_skip_connection(temporal_out) + spatial_out)

        out = self.last_layer(out)
        out = self.dropout(out)
        return self.relu4(out + self.crop_skip_connection(x))


class P3DC(P3DBlock):
    def forward(self, x: torch.Tensor):
        # x shape (N, F, T, Lat, Lon)
        out = self.first_layer(x)
        out = self.relu1(out)

        out = self.spatial_conv(out)
        out = self.relu2(out)

        temporal_out = self.temporal_conv(out)
        out = self.relu3(out + temporal_out)

        out = self.last_layer(out)
        out = self.dropout(out)
        return self.relu4(out + self.crop_skip_connection(x))
