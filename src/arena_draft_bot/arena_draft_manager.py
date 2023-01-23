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
import re
import torch

from arena_draft_bot.draft_manager import DraftManager

from set_config import SetConfig


class ArenaLogLineParser:
    """
    Parser that turns arena log lines into vectorized packs and picks.

    Can handle "pack lines" and "pick lines", which contain information about
    what cards were shown in the draft and what the user picked as lists of
    arena ids.

    Note that Arena does not provide a formal API, so this could break on future
    versions if the internal log format changes. This is currently dependent on
    the log format as of Jan 2023.
    """
    def __init__(self, sorted_card_names, arena_id_map) -> None:
        self.sorted_card_names = sorted_card_names
        self.name_index_map = {c:i for i,c in enumerate(sorted_card_names)}
        self.arena_id_map = arena_id_map
    
    def _names_to_vector(self, names):
        vec = torch.zeros(len(self.sorted_card_names), dtype=torch.float32)
        for name in names:
            vec[self.name_index_map[name]] += 1.0
        return vec

    def pack_line_to_vector(self, line):
        match = re.search("Draft.Notify (?P<json_blob>\{.*\})", line)
        if match:
            json_blob = match.group("json_blob")
            draft_obj = json.loads(json_blob)
            pack_card_names = []
            for id in draft_obj["PackCards"].split(","):
                try:
                    pack_card_names.append(self.arena_id_map[id])
                except:
                    print("failed to get pack card for arena id ", id)
            return self._names_to_vector(pack_card_names)
        else:
            print("no match on pack line: ", line)
        return torch.zeros(len(self.sorted_card_names), dtype=torch.float32)


    def pick_line_to_vector(self, line):
        match = re.search("Event_PlayerDraftMakePick (?P<json_blob>\{.*\})", line)
        if match:
            json_blob = match.group("json_blob")
            draft_obj = json.loads(json_blob)
            try:
                draft_req = json.loads(draft_obj["request"])
                payload = json.loads(draft_req["Payload"])
                id = str(payload["GrpId"])
                name = ""
                try:
                    name = self.arena_id_map[id]
                except:
                    print("failed to get pick card for arena id", payload["GrpId"])
                if name:
                    return self._names_to_vector([name])                
            except:
                print("failed to parse pick line json")
        else:
            print("no match on pick line: ", line)
        return torch.zeros(len(self.sorted_card_names), dtype=torch.float32)


class ArenaDraftManager(DraftManager):
    """
    DraftManager for handling Arena drafts from logs.

    Log line parsing logic is done by ArenaLogLineParser above.
    """
    def __init__(self, per_set_config: SetConfig, model_file, draft_uid) -> None:
        super().__init__(per_set_config, model_file, draft_uid)
        sorted_card_names = json.load(open(per_set_config.card_list_path, "r"))
        arena_id_map = json.load(open(per_set_config.card_arena_ids_path, "r"))
        self.line_parser = ArenaLogLineParser(sorted_card_names, arena_id_map)
    
    def handle_pack_line(self, line):
        self._handle_draft_pack(self.line_parser.pack_line_to_vector(line))

    def handle_pick_line(self, line):
        self._handle_human_pick(self.line_parser.pick_line_to_vector(line))
