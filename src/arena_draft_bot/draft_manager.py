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

import torch

from set_config import SetConfig
from arena_draft_bot.pick_display_utils import PickDisplayer


class DraftManager:
    """
    Base class for running the model on draft packs. 
    
    Uses PickDisplayer to show the output to the user. 
    
    This class should be overridden to convert the draft data format (e.g.
    arena or MTGO logs) into a form that this class can understand (vectorized
    packs and picks). The subclass can call _handle_draft_pack and
    _handle_draft_pick after vectorizing the data.
    """
    def __init__(self, per_set_config: SetConfig, model_file, draft_uid, plot_picks=False) -> None:
        self.draft_net = torch.load(model_file)
        self.set_size = per_set_config.set_size
        self.set_id = per_set_config.set_id
        self.pool_vector = torch.zeros(per_set_config.set_size, dtype=torch.float32)
        self.pick_displayer = PickDisplayer(per_set_config, draft_uid)
        self.plot_picks = plot_picks

    def _handle_draft_pack(self, pack_vector):
        self.draft_net.eval()
        with torch.no_grad():
            # The model was trained on 2D batches, so the input needs to be
            # converted to a 2D tensor.
            inputs = torch.zeros(1, self.set_size*2, dtype=torch.float32)
            inputs[0] = torch.cat((pack_vector, self.pool_vector))
            preds = self.draft_net(inputs)
            self.pick_displayer.print_pack_and_predictions_to_console(pack_vector, preds)
            if self.plot_picks:
                self.pick_displayer.plot_predictions(pack_vector, preds)
    
    def _handle_human_pick(self, pick_vector):
        # Aggregate the human's real picks to form the pool.
        self.pool_vector += pick_vector
        self.pick_displayer.print_current_pool_to_console(self.pool_vector)
