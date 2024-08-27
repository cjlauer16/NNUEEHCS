import torch.nn
import collections
import io
import yaml
from .models import deltaUQ_MLP, EnsembleModel
import copy
import types

class LayerBuilder(object):
    # adapted from https://gist.github.com/ferrine/89d739e80712f5549e44b2c2435979ef
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_network(architecture, builder=LayerBuilder(torch.nn.__dict__)):
    """
    Configuration for feedforward networks is list by nature. We can write 
    this in simple data structures. In yaml format it can look like:
    .. code-block:: yaml
        architecture:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2
    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:
    .. code-block:: python
        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})
    the simpliest ever builder is just the following function:
    .. code-block:: python
         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)
    
    Some more advanced builders catch exceptions and format them in debuggable way or merge 
    namespaces for name lookup
    
    .. code-block:: python
    
        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(architecture, builder=extended_builder)
        
    """
    layers = []
    architecture = copy.deepcopy(architecture)
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)


class InfoGrabbBase:
    def __init__(self, descr):
        self.descr = descr

    def num_layers(self):
        return len(self.descr)


class CNNInfoGrabber(InfoGrabbBase):
    def __init__(self, descr):
        super().__init__(descr)

    def is_cnn(self):
        return True

    def is_mlp(self):
        return False

    def num_inputs(self):
        return self.descr[0]['Conv2d']['args'][0]

    def set_num_inputs(self, num_inputs):
        self.descr[0]['Conv2d']['args'][0] = num_inputs


class MLPInfoGrabber(InfoGrabbBase):
    def __init__(self, descr):
        super().__init__(descr)

    def is_mlp(self):
        return True

    def is_cnn(self):
        return False

    def num_inputs(self):
        return self.descr[0]['Linear']['args'][0]

    def set_num_inputs(self, num_inputs):
        self.descr[0]['Linear']['args'][0] = num_inputs


class ModelInfo:
    def __init__(self):
        pass

    @classmethod
    def get_info_grabber(cls, model_descr):
        if 'Conv2d' in model_descr[0]:
            return CNNInfoGrabber(model_descr)
        else:
            return MLPInfoGrabber(model_descr)


class ModelBuilder:
    def __init__(self, model_descr):
        self.model_descr = copy.deepcopy(model_descr)

    def build(self):
        built = build_network(self.model_descr)
        return built

    def update_info(self, info):
        return info

    def get_info(self):
        info = ModelInfo.get_info_grabber(self.model_descr)
        self.update_info(info)
        return info


class DeltaUQMLPModelBuilder(ModelBuilder):
    def __init__(self, base_descr, duq_descr):
        super().__init__(base_descr)
        self.duq_descr = duq_descr

    def build(self):
        base_model = super().build()
        return deltaUQ_MLP(base_model, estimator=self.duq_descr['estimator'])

    def update_info(self, info):
        estimator = self.duq_descr['estimator']

        def get_estimator(self):
            return estimator
        info.set_num_inputs(2 * info.num_inputs())
        info.get_estimator = types.MethodType(get_estimator, info)


class PAGERModelBuilder(DeltaUQMLPModelBuilder):
    # for now, we will just inherit from DUQ.
    # Later update as needed
    def __init__(self, base_descr, duq_descr):
        super().__init__(base_descr, duq_descr)


class EnsembleModelBuilder(ModelBuilder):
    def __init__(self, base_descr, ensemble_descr):
        super().__init__(base_descr)
        self.ensemble_descr = ensemble_descr

    def build(self):
        info = self.get_info()
        build = super().build
        base_models = [build() for _ in range(info.get_num_models())]
        return EnsembleModel(base_models)

    def update_info(self, info):
        num_models = self.ensemble_descr['num_models']

        def get_num_models(self):
            return num_models
        info.get_num_models = types.MethodType(get_num_models, info)


class KDEModelBuilder(ModelBuilder):
    def __init__(self, base_descr, kde_descr):
        super().__init__(base_descr)
