import torch.nn as nn
from abc import ABC, abstractmethod


# abstract model class; any proper subclass from ParentModel will work with train_with_tmp_files.py
class ParentModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        # must return data_x, data_y
        pass

    @property
    @abstractmethod
    def lookback(self):
        pass

    @property
    @abstractmethod
    def lead_time(self):
        pass

    @property
    @abstractmethod
    def swh_reduce(self):
        pass

    @property
    @abstractmethod
    def spatial_reduction(self):
        pass

    @property
    @abstractmethod
    def loss_function(self):
        pass

    @property
    @abstractmethod
    def optimiser(self):
        pass

    @property
    @abstractmethod
    def scheduler(self):
        pass

    @property
    @abstractmethod
    def crop_output(self):
        pass


class OldParentModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def get_lookback(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        # must return data_x, data_y
        pass

    @abstractmethod
    def get_loss_function(self):
        return self.loss_function

    @abstractmethod
    def get_optimiser(self):
        # must return optimiser, scheduler
        # return optimiser, None if there is no scheduler
        pass
