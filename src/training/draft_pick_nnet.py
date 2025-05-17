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


class DraftPickNN(nn.Module):
    """
    NNet for picking cards in a draft. Based on the techniques used in
    https://arxiv.org/pdf/2009.00655.pdf.

    There are 4 linear layers, each of size N (size of the card pool). The
    pack is used to constrain the output to possible picks.
    """

    def __init__(self, card_pool_size):
        """
        INPUTS:
        - card_pool_size: The size of the card pool for the set. This will be used
            as the dimension for each NN layer, as well as the input / output size.
        """
        super().__init__()
        self.net_dim = card_pool_size

        self.ns = 0.01

        # Define the functions that will be used to run the net.
        self.bn = nn.BatchNorm1d(self.net_dim)
        self.linear1 = nn.Linear(self.net_dim, self.net_dim)
        self.bn1 = nn.BatchNorm1d(self.net_dim)
        self.relu1 = nn.LeakyReLU(negative_slope = self.ns)
        self.dropout1 = nn.Dropout(0.5)
        
        self.linear2 = nn.Linear(self.net_dim, self.net_dim)
        self.bn2 = nn.BatchNorm1d(self.net_dim)
        self.relu2 = nn.LeakyReLU(negative_slope = self.ns)
        self.dropout2 = nn.Dropout(0.5)
        
        self.linear3 = nn.Linear(self.net_dim, self.net_dim)
        self.bn3 = nn.BatchNorm1d(self.net_dim)
        self.relu3 = nn.LeakyReLU(negative_slope = self.ns)
        self.dropout3 = nn.Dropout(0.5)
        
        self.linear4 = nn.Linear(self.net_dim, self.net_dim)

        # Cache the most recent activations of each hidden layer for use in training
        # SAEs on this model.
        self.activations = {
            'layer1': None,
            'layer2': None,
            'layer3': None,
        }
    
    def get_most_recent_hidden_activations(self):
        """
        Returns the most recent activations of each hidden layer. This is used
        for training SAEs on this model.
        """
        return self.activations

    def forward(self, x):
        """
        INPUTS:
        - x: A vectorized representation of the pack and the player's existing
            card pool. This is a vector of length (2 * net_dim), where the first
            net_dim elements are the pack and the last net_dim elements are the
            pool.
        
        As described in the paper above, we apply batch norm after each linear
        layer (normalize ranges of inputs), as well as 0.5 dropout to prevent
        overfitting.

        The network predicts the best cards to pick, and then the "in pack" constraint
        is applied at the end.
        """
        pack = x[:, :self.net_dim]
        pool = x[:, self.net_dim:]

        y = self.linear1(pool)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        if not self.training:
            self.activations['layer1'] = y.detach()

        y = self.linear2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.dropout2(y)
        if not self.training:
            self.activations['layer2'] = y.detach()

        y = self.linear3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.dropout3(y)
        if not self.training:
            self.activations['layer3'] = y.detach()

        y = self.linear4(y)

        y *= pack
        return y
