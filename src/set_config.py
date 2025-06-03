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
#



# Functions for getting the important parameters for a particular set.
#
# See set_config.json for format. Should return a parsed json object of the form:
# {
#     "set_id": "BRO",
#     "model_path": "../data/BRO/models/draft_bot_nnet_2023-01-03-20-epoch-adam",
#     "card_list_path": "../data/BRO/per_card_data/BRO_canonical_card_list.json",
#     "card_arena_ids_path": "../data/BRO/per_card_data/arena_ids_to_names.json",
#     "card_image_dir": "../data/BRO/card_images",
#     "card_metadata_dir": "../data/BRO/card_metadata",
#     "set_size": 335
# }
import json
import os

from dataclasses import dataclass


def get_set_config(set_id):
    return get_set_config_from_path(os.path.join("../data/", set_id, "set_config.json"))


def get_set_config_from_path(path):
    config = json.load(open(path, "r"))
    return SetConfig(
        config["set_id"],
        config["model_path"],
        config["card_list_path"],
        config["card_arena_ids_path"],
        config["card_image_dir"],
        config["card_metadata_dir"],
        config["set_size"]
    )


@dataclass
class SetConfig:
    set_id: str
    model_path: str
    card_list_path: str
    card_arena_ids_path: str
    card_image_dir: str
    card_metadata_dir: str
    set_size: int
