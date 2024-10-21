from train_triple import train_triple
from test_triple import test_triple


__factory__ = {
    'train_triple': train_triple,
    'test_triple': test_triple,
}

def build_handler(phase, model):
    key_handler = '{}_{}'.format(phase, model)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]