import pytest
from nnueehcs.model_builder import (build_network, ModelBuilder,
                                    DeltaUQMLPModelBuilder,
                                    EnsembleModelBuilder,
                                    PAGERModelBuilder,
                                    KDEModelBuilder,
                                    KDEMLPModel,
                                    MCDropoutModelBuilder,
                                    ChecksumModelBuilder)
import torch
import io
import yaml
import os
from torch import nn
from nnueehcs.deltauq import deltaUQ_MLP, deltaUQ_CNN

def assert_models_equal(model1, model2):
    for layer1, layer2 in zip(model1.children(), model2.children()):
        assert type(layer1) == type(layer2), f"Layer types differ: {type(layer1)} != {type(layer2)}"
        
        if isinstance(layer1, nn.Conv2d):
            assert layer1.in_channels == layer2.in_channels, f"In channels differ: {layer1.in_channels} != {layer2.in_channels}"
            assert layer1.out_channels == layer2.out_channels, f"Out channels differ: {layer1.out_channels} != {layer2.out_channels}"
            assert layer1.kernel_size == layer2.kernel_size, f"Kernel sizes differ: {layer1.kernel_size} != {layer2.kernel_size}"
            assert layer1.stride == layer2.stride, f"Strides differ: {layer1.stride} != {layer2.stride}"
            assert layer1.padding == layer2.padding, f"Padding differs: {layer1.padding} != {layer2.padding}"
        
        elif isinstance(layer1, nn.Linear):
            assert layer1.in_features == layer2.in_features, f"In features differ: {layer1.in_features} != {layer2.in_features}"
            assert layer1.out_features == layer2.out_features, f"Out features differ: {layer1.out_features} != {layer2.out_features}"
        
        elif isinstance(layer1, nn.BatchNorm2d):
            assert layer1.num_features == layer2.num_features, f"Num features differ: {layer1.num_features} != {layer2.num_features}"
        
        elif isinstance(layer1, (nn.ReLU, nn.Dropout)):
            assert layer1.inplace == layer2.inplace, f"Inplace flag differs: {layer1.inplace} != {layer2.inplace}"
        
        if isinstance(layer1, nn.Dropout):
            assert layer1.p == layer2.p, f"Dropout probability differs: {layer1.p} != {layer2.p}"

@pytest.fixture()
def architecture1():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=25, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 25, kernel_size=5, stride=1, padding='same')
    )
    return model

@pytest.fixture()
def architecture2():
    model = nn.Sequential(
        nn.Linear(16, 25),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(25),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(25, 5)
    )
    return model


@pytest.fixture()
def kde_architecture1(architecture1):
    return KDEMLPModel(architecture1)


@pytest.fixture()
def kde_architecture2(architecture2):
    return KDEMLPModel(architecture2)


@pytest.fixture()
def duq_architecture1():
    model = nn.Sequential(
        nn.Conv2d(6, 16, kernel_size=25, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 25, kernel_size=5, stride=1, padding='same')
    )
    return deltaUQ_CNN(model)


@pytest.fixture()
def duq_architecture2():
    model = nn.Sequential(
        nn.Linear(32, 25),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(25),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(25, 5)
    )
    return deltaUQ_MLP(model)

@pytest.fixture()
def model_descr_yaml():
    model_yaml = """
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
        padding: same

architecture2:
    - Linear:
        args: [16, 25]
    - ReLU:
        inplace: true
    - BatchNorm2d:
        args: [25]
    - Dropout:
        args: [0.2]
        inplace: True
    - Linear:
        args: [25, 5]

architecture3:
    - Linear:
        args: [16, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 5]
delta_uq_model:
    estimator: std
    num_anchors: 2
mc_dropout_model:
    num_samples: 10
    dropout_percent: 0.2
pager_model:
    estimator: std
    num_anchors: 3
kde_model:
    bandwidth: scott
ensemble_model:
    num_models: 10
checksum_model:
    n_checksums: 1
    checksum_name: sine
    freq: 1
    checksum_pred_weight: 1
    checksum_penalty_weight: 2
    checksum_reward_weight: 3
    oos_min: 0.2
    oos_max: 0.7
"""
    return model_yaml


def test_build_network(model_descr_yaml, architecture1, architecture2):

    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    
    # Test architecture
    actual_model = build_network(model_descr['architecture'])
    assert_models_equal(actual_model, architecture1)

    # Test architecture2
    actual_model2 = build_network(model_descr['architecture2'])
    assert_models_equal(actual_model2, architecture2)


def test_model_builder(model_descr_yaml, architecture1, architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = ModelBuilder(model_descr['architecture'])
    arch1 = model_builder.build()
    assert_models_equal(arch1, architecture1)
    builder2 = ModelBuilder(model_descr['architecture2'])
    arch2 = builder2.build()
    assert_models_equal(arch2, architecture2)

    info = model_builder.get_info()
    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    info2 = builder2.get_info()
    assert info2.is_cnn() == False
    assert info2.is_mlp() == True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info, 'get_estimator')

@pytest.mark.skip(reason="This test is currently failing")
def test_duq_model_builder(model_descr_yaml, duq_architecture1, duq_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = DeltaUQMLPModelBuilder(model_descr['architecture'], model_descr['delta_uq_model'])
    info = model_builder.get_info()
    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 6
    assert info.get_estimator() == 'std'

    net = model_builder.build()
    assert_models_equal(net, duq_architecture1)
    assert net.num_anchors == 2

    model_builder = DeltaUQMLPModelBuilder(model_descr['architecture2'], model_descr['delta_uq_model'])
    info = model_builder.get_info()
    assert info.is_cnn() == False
    assert info.is_mlp() == True
    assert info.num_layers() == 5
    assert info.num_inputs() == 32
    assert info.get_estimator() == 'std'

    net = model_builder.build()
    assert_models_equal(net, duq_architecture2)

@pytest.mark.skip(reason="This test is currently failing")
def test_pager_model_builder(model_descr_yaml, duq_architecture1, duq_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = PAGERModelBuilder(model_descr['architecture'], model_descr['pager_model'])
    net = model_builder.build()
    assert net.num_anchors == 3
    assert net.num_anchors == 3
    assert_models_equal(net, duq_architecture1)
    info = model_builder.get_info()
    assert info.is_cnn() is True
    assert info.is_mlp() is False
    assert info.num_layers() == 3
    assert info.num_inputs() == 6
    assert info.get_estimator() == 'std'

    model_builder2 = PAGERModelBuilder(model_descr['architecture2'], model_descr['pager_model'])
    net2 = model_builder2.build()
    assert_models_equal(net2, duq_architecture2)
    info2 = model_builder2.get_info()
    assert info2.is_cnn() is False
    assert info2.is_mlp() is True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 32
    assert info2.get_estimator() == 'std'


def test_ensemble_model_builder(model_descr_yaml):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = EnsembleModelBuilder(model_descr['architecture'], model_descr['ensemble_model'])
    ensemble = model_builder.build()
    info = model_builder.get_info()
    assert info.get_num_models() == 10

    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    assert not hasattr(info, 'get_estimator')

    model_builder2 = EnsembleModelBuilder(model_descr['architecture2'], model_descr['ensemble_model'])
    ensemble2 = model_builder2.build()
    info2 = model_builder2.get_info()
    assert info2.get_num_models() == 10

    assert info2.is_cnn() == False
    assert info2.is_mlp() == True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info2, 'get_estimator')


def test_kde_model_builder(model_descr_yaml, kde_architecture1, kde_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = KDEModelBuilder(model_descr['architecture'],
                                    model_descr['kde_model'])
    arch1 = model_builder.build()
    assert_models_equal(arch1, kde_architecture1)
    builder2 = KDEModelBuilder(model_descr['architecture2'],
                               model_descr['kde_model'])
    arch2 = builder2.build()
    assert_models_equal(arch2, kde_architecture2)

    info = model_builder.get_info()
    assert info.is_cnn() is True
    assert info.is_mlp() is False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    info2 = builder2.get_info()
    assert info2.is_cnn() is False
    assert info2.is_mlp() is True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info, 'get_estimator')

def test_mc_model_builder(model_descr_yaml, architecture1, architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = MCDropoutModelBuilder(model_descr['architecture3'],
                                          model_descr['mc_dropout_model'])
    arch1 = model_builder.build()
    expected = """
MCDropoutModel(
  (model): Sequential(
    (0): Linear(in_features=16, out_features=25, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=25, out_features=25, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=25, out_features=25, bias=True)
    (7): ReLU(inplace=True)
    (8): Linear(in_features=25, out_features=5, bias=True)
  )
)
"""
    assert str(arch1).strip() == expected.strip()
    arch1.eval()
    for layer in arch1.model:
        if isinstance(layer, nn.Dropout):
            assert layer.training == True
        else:
            assert layer.training == False
    arch1.train()
    for layer in arch1.model:
        assert layer.training == True

    model_builder = MCDropoutModelBuilder(model_descr['architecture2'],
                                          model_descr['mc_dropout_model'])
    arch2 = model_builder.build()
    arch2.eval()
    for layer in arch2.model:
        if isinstance(layer, nn.Dropout):
            assert layer.training == True
        else:
            assert layer.training == False
    arch2.train()
    for layer in arch2.model:
        assert layer.training == True

def test_checksum_model_builder(model_descr_yaml):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    # model_builder = ChecksumModelBuilder(model_descr['architecture'], model_descr['checksum_model'])
    # checksum_model = model_builder.build()
    # info = model_builder.get_info()
    # assert info.get_num_outputs() == 10

    # assert info.is_cnn() == True
    # assert info.is_mlp() == False
    # assert info.num_layers() == 3
    # assert info.num_inputs() == 3

    model_builder3 = ChecksumModelBuilder(model_descr['architecture3'], model_descr['checksum_model'])
    checksum_model = model_builder3.build()
    info3 = model_builder3.get_info()
    assert info3.num_outputs() == 6

    assert info3.is_cnn() == False
    assert info3.is_mlp() == True
    assert info3.num_layers() == 7
    assert info3.num_inputs() == 16
    assert checksum_model.n_checksums == 1
    assert checksum_model.checksum_name == 'sine'
    assert checksum_model.checksum_pred_weight == 1
    assert checksum_model.checksum_penalty_weight == 2
    assert checksum_model.checksum_reward_weight == 3
    assert checksum_model.oos_min == 0.2
    assert checksum_model.oos_max == 0.7


    assert not hasattr(info3, 'get_estimator')