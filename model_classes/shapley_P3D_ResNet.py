import torch
from .P3D_ResNet import P3DResNet



class ShapleyP3DResNet(P3DResNet):
    def forward(self, x):
        # print('in:', x[-1, :, 0, -1, -1])
        # x shape is (N, F, T, Lat, Lon)
        if self.input_norm:
            feature_masks = []
            is_feature_masked_list = []
            for feature in range(x.shape[1]):
                # use -99,999 as substitute for NaN; NaN messes with the tensor operations
                mask = x[:, feature] == -99_999
                feature_masks.append(mask)
                if torch.sum(mask) >= 1:  # there are NaNs (i.e. -99_999 values present)
                    is_feature_masked_list.append(True)
                else:
                    is_feature_masked_list.append(False)
            maxima_tensor = torch.as_tensor(self.maxima).repeat((x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
            minima_tensor = torch.as_tensor(self.minima).repeat((x.shape[0], x.shape[4], x.shape[2], x.shape[3], 1)).transpose(1, 4)
            x = (x - minima_tensor) / (maxima_tensor - minima_tensor)
            for feature, mask, is_feature_masked in zip(range(x.shape[1]), feature_masks, is_feature_masked_list):
                if not is_feature_masked:
                    continue

                # set out of coalition features to -1 for sea and 0 for land
                x[:, feature] = torch.where(mask, -1, 0)
        # print('out:', x[0, :, 0, -1, -1], '\n')

        for block in self.blocks:
            x = block(x)

        # final layer to squish feature and time dimensions
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = x.permute((0, 2, 3, 1))
        for layer in self.final_layers:
            x = layer(x)

        x = x.permute(0, 3, 1, 2)

        if self.input_norm:
            minima = torch.full_like(x, self.minima[-1])  # assumes target feature is always supplied-last to model
            maxima = torch.full_like(x, self.maxima[-1])
            x = x * (maxima - minima) + minima
        return x
