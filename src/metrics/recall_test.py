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

from metrics.recall import recall_at_n_overall, recall_at_n_per_pick


def test_recall_at_n_overall_2d():
    preds = torch.Tensor(
        [[1,2,3],
        [3,2,1]]
    )
    labels = torch.Tensor(
        [[1,0,0],
        [1,0,0]]
    )
    # Argmax matches on second example but not the first.
    assert recall_at_n_overall(preds, labels, n=1) == 0.5
    # Top 2 matches on second example but not the first.
    assert recall_at_n_overall(preds, labels, n=2) == 0.5
    # Top 3 matches for both examples.
    assert recall_at_n_overall(preds, labels, n=3) == 1.0


def test_recall_at_n_overall_3d():
    # Same test as 2d, but adding a draft pick dimension that should be
    # unrolled internally.
    preds = torch.Tensor(
        [[[1,2,3]],
        [[3,2,1]]]
    )
    labels = torch.Tensor(
        [[[1,0,0]],
        [[1,0,0]]]
    )
    # Argmax matches on second example but not the first.
    assert recall_at_n_overall(preds, labels, n=1) == 0.5
    # Top 2 matches on second example but not the first.
    assert recall_at_n_overall(preds, labels, n=2) == 0.5
    # Top 3 matches for both examples.
    assert recall_at_n_overall(preds, labels, n=3) == 1.0



def test_recall_at_n_per_pick():
    # Repeat examples across the "pick" dimension. We should get the same
    # recall numbers per pick.
    preds = torch.Tensor(
        [[[1,2,3], [1,2,3]],
        [[3,2,1], [3,2,1]]]
    )
    labels = torch.Tensor(
        [[[1,0,0], [1,0,0]],
        [[1,0,0], [1,0,0]]]
    )

    # Argmax matches on second example but not the first.
    matches = recall_at_n_per_pick(preds, labels, n=1) == torch.Tensor([0.5, 0.5])
    assert matches.all()
    # Top 2 matches on second example but not the first.
    matches = recall_at_n_per_pick(preds, labels, n=2) == torch.Tensor([0.5, 0.5])
    assert matches.all()
    # Top 3 matches for both examples.
    matches = recall_at_n_per_pick(preds, labels, n=3) == torch.Tensor([1.0, 1.0])
    assert matches.all()