#!/usr/bin/python
#
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from training.draft_pick_nnet import DraftPickNN
from set_config import get_set_config

@pytest.fixture
def set_config_for_test():
    return get_set_config("BRO")

def test_draft_pick_nnet_init(set_config_for_test):
    net = DraftPickNN(set_config_for_test.set_size)

    # Check that the network layers are initialized correctly
    assert net.net_dim == set_config_for_test.set_size
    assert isinstance(net.linear1, torch.nn.Linear)
    assert isinstance(net.bn1, torch.nn.BatchNorm1d)
    assert isinstance(net.relu1, torch.nn.LeakyReLU)
    assert isinstance(net.dropout1, torch.nn.Dropout)
    assert net.get_most_recent_hidden_activations() == {
        "layer1": None,
        "layer2": None,
        "layer3": None,
    }


def test_draft_pick_nnet_forward_pass(set_config_for_test):
    net = DraftPickNN(set_config_for_test.set_size)
    net.eval()

    # Input is 2x the net dim because there's a pack and a pool.
    sample_input = torch.randint(0, 2, (1, set_config_for_test.set_size * 2)).float()

    output = net(sample_input)
    assert output.shape == (1, set_config_for_test.set_size)
    assert net.get_most_recent_hidden_activations()["layer1"].shape == (1, set_config_for_test.set_size)
    assert net.get_most_recent_hidden_activations()["layer2"].shape == (1, set_config_for_test.set_size)
    assert net.get_most_recent_hidden_activations()["layer3"].shape == (1, set_config_for_test.set_size)
