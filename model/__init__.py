from .triple import *

__factory__ = {
    'triple': triple_model
}

def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknow model:",name)
    return __factory__[name](*args,**kwargs)