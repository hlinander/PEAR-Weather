import torch

from experiments.weather.models.pangu import Pangu, PanguConfig
from experiments.weather.models.pangu import PanguParametrized, PanguParametrizedConfig

from experiments.weather.models.swin_hp_pangu_pad import SwinHPPanguPadConfig
from experiments.weather.models.swin_hp_pangu_pad import SwinHPPanguPad


class _ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[SwinHPPanguPadConfig.__name__] = SwinHPPanguPad
        self.models[PanguConfig.__name__] = Pangu
        self.models[PanguParametrizedConfig.__name__] = PanguParametrized

    def register(self, config_class, model_class):
        self.models[config_class.__name__] = model_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__.__name__](model_config, data_config)

    def get_class(self, model_config):
        return self.models[model_config.__class__.__name__]


_model_factory = None


def get_factory():
    global _model_factory
    if _model_factory is None:
        _model_factory = _ModelFactory()

    return _model_factory
