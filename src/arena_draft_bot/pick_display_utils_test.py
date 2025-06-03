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

from arena_draft_bot.pick_display_utils import PickDisplayer
from set_config import get_set_config

@pytest.fixture
def set_config_for_test():
    return get_set_config("BRO")

def test_print_pool(capsys, set_config_for_test):
    pick_displayer = PickDisplayer(set_config_for_test, draft_uid="test")

    pool = torch.zeros(set_config_for_test.set_size)
    # Add one copy of the first card in BRO, Adaptive Automaton.
    pool[0] = 1.0
    # Add one copy of the second card in BRO, Aeronaut Cavalry.
    pool[1] = 1.0
    pick_displayer.print_current_pool_to_console(pool)
    captured = capsys.readouterr()
    assert "Adaptive Automaton" in captured.out
    assert "Aeronaut Cavalry" in captured.out

def test_print_pack_and_preds(capsys, set_config_for_test):
    pick_displayer = PickDisplayer(set_config_for_test, draft_uid="test")

    pack = torch.zeros(set_config_for_test.set_size)
    preds = torch.zeros(1, set_config_for_test.set_size)
    # Add one copy of the first card in BRO, Adaptive Automaton.
    pack[0] = 1.0
    preds[0][0] = 15.0
    # Add one copy of the second card in BRO, Aeronaut Cavalry.
    pack[1] = 1.0
    preds[0][1] = 20.0
    pick_displayer.print_pack_and_predictions_to_console(pack, preds)
    captured = capsys.readouterr()
    assert "Adaptive Automaton" in captured.out
    assert "Aeronaut Cavalry" in captured.out
    # Prints scores.
    assert "('Aeronaut Cavalry', 20.0" in captured.out
    assert "('Adaptive Automaton', 15.0" in captured.out

def test_get_card_metadata_for_input_vector(set_config_for_test):
    pick_displayer = PickDisplayer(set_config_for_test, draft_uid="test")

    input_vector = torch.zeros(set_config_for_test.set_size)
    input_vector[0] = 1.0
    input_vector[1] = 1.0
    metadata = pick_displayer.get_card_metadata_for_input_vector(input_vector)
    assert metadata[0]["name"] == "Adaptive Automaton"
    assert metadata[1]["name"] == "Aeronaut Cavalry"