from .dataset import Dataset

__factory__ = {
    'triple': Dataset,
}


def build_dataset(model_type, cfg):
    if model_type not in __factory__:
        raise KeyError("Unknown Dataset type:", model_type)
    return __factory__[model_type](cfg, cfg.phase)
