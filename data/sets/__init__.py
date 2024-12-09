from data.sets.cifarfs import CIFARFS
from data.sets.miniImageNet import miniImageNet
from data.sets.minicifar import minicifar

__imgfewshot_factory = {
        'co_cifarfs': CIFARFS,
        'co_miniimagenet': miniImageNet,
        'un_cifarfs': CIFARFS,
        'un_miniimagenet': miniImageNet,
        'cifar_mini': minicifar,
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](name, **kwargs)

