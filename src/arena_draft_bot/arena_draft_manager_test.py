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
import json
import pytest
import torch

from arena_draft_bot.arena_draft_manager import ArenaLogLineParser
from set_config import get_set_config

@pytest.fixture
def set_info_for_test():
    bro_config = get_set_config("BRO")
    sorted_card_names = json.load(open(bro_config.card_list_path, "r"))
    arena_id_map = json.load(open(bro_config.card_arena_ids_path, "r"))
    return (sorted_card_names, arena_id_map)

def test_successful_pick_line(set_info_for_test):
    sorted_card_names, arena_id_map = set_info_for_test
    parser = ArenaLogLineParser(sorted_card_names, arena_id_map)
    pick_line = (
        r'[171539] [UnityCrossThreadLogger]==> ' + 
        r'Event_PlayerDraftMakePick {"id":"2fea72fc-43c5-4c27-844e-c022cece5609",' + 
        r'"request":"{\"Type\":620,\"TransId\":\"2fea72fc-43c5-4c27-844e-c022cece5609\",' + 
        r'\"Payload\":\"{\\\"DraftId\\\":\\\"908c0905-b9eb-42bd-bd65-1f8250fd54f1\\\",\\\"GrpId\\\":82614,\\\"Pack\\\":1,\\\"Pick\\\":1}\"}"}'
    )
    pick_vector = parser.pick_line_to_vector(pick_line)
    # A pick vector contains exactly one entry.
    assert torch.sum(pick_vector).item() == 1.0

def test_returns_empty_pick_if_no_match(set_info_for_test):
    sorted_card_names, arena_id_map = set_info_for_test
    parser = ArenaLogLineParser(sorted_card_names, arena_id_map)
    bad_pick_line = r'[171545] [UnityCrossThreadLogger] '
    pick_vector = parser.pick_line_to_vector(bad_pick_line)
    assert torch.sum(pick_vector).item() == 0.0

def test_successful_pack_line(set_info_for_test):
    sorted_card_names, arena_id_map = set_info_for_test
    parser = ArenaLogLineParser(sorted_card_names, arena_id_map)
    pack_line = (
        r'[171545] [UnityCrossThreadLogger]Draft.Notify ' + 
        r'{"draftId":"908c0905-b9eb-42bd-bd65-1f8250fd54f1","SelfPick":2,"SelfPack":1,' +
        r'"PackCards":"82575,82741,82669,82515"}'
    )
    pack_vector = parser.pack_line_to_vector(pack_line)
    # Should be 4 cards in the vector to match PackCards above.
    assert torch.sum(pack_vector).item() == 4.0

def test_returns_empty_pack_if_no_match(set_info_for_test):
    sorted_card_names, arena_id_map = set_info_for_test
    parser = ArenaLogLineParser(sorted_card_names, arena_id_map)
    bad_pack_line = r'[171545] [UnityCrossThreadLogger] '
    pack_vector = parser.pack_line_to_vector(bad_pack_line)
    assert torch.sum(pack_vector).item() == 0.0
