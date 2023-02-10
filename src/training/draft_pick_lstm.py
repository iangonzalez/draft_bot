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
import torch.nn as nn
import torch.nn.functional as F

class DraftPickLSTM(nn.Module):
    """
    LSTM for picking cards in a draft.

    3 stacked LSTMs followed by a final linear layer + softmax, as well as the
    in-pack constraint.
    """
    def __init__(self, card_pool_size):
        """
        INPUTS:
        - card_pool_size: The size of the card pool for the set. This will be used
            as the dimension for the LSTM hidden layer, as well as the input / output size.
        """
        super().__init__()
        self.set_size_dim = card_pool_size
        self.hidden_layer_dim = card_pool_size

        self.lstm = nn.LSTM(input_size=self.set_size_dim * 2, hidden_size=self.hidden_layer_dim,
                            num_layers=3, dropout=0.5)
        self.final_linear = nn.Linear(self.hidden_layer_dim, self.set_size_dim)

    def forward(self, draft_sequence):
        pack = draft_sequence[:, :, :self.set_size_dim]
        lstm_out, _ = self.lstm(draft_sequence)
        lstm_out = self.final_linear(lstm_out)
        # Apply card in pack constraint.
        lstm_out *= pack
        # Softmax over the classes (the final dimension).
        lstm_out = F.log_softmax(lstm_out, dim=2)
        return lstm_out
