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


def recall_at_n_per_pick(preds, labels, n=1):
    """
    Computes Recall@N for each pick for a given N.

    A prediction is correct for the purposes of this metric if the top N
    ranked outputs contain the label.

    Assumes 3d input tensors with dimensions (draft number, pick number,
    card).
    """
    highest_predicted = torch.topk(preds, n, dim=2, sorted=True)
    actual_picked = torch.argmax(labels, 2)
    correct = None
    for i in range(n):
        pred_picks = highest_predicted.indices[:, :, i]
        if correct is None:
            correct = pred_picks == actual_picked
        else:
            correct |= pred_picks == actual_picked
    recall = correct.sum(dim=0) / correct.shape[0]
    return recall


def recall_at_n_overall(preds, labels, n=1):
    """
    Computes overall Recall@N.

    A prediction is correct for the purposes of this metric if the top N
    ranked outputs contain the label.

    Assumes 2d input tensors with dimensions (pick number, card). If input is
    3d the picks dimension will be unrolled.
    """
    if len(preds.shape) == 3 and len(labels.shape) == 3:
        preds = preds.reshape((preds.shape[0] * preds.shape[1], preds.shape[2]))
        labels = labels.reshape((labels.shape[0] * labels.shape[1], labels.shape[2]))
    highest_predicted = torch.topk(preds, n, dim=1, sorted=True)
    actual_picked = torch.argmax(labels, 1)
    correct = None
    for i in range(n):
        pred_picks = highest_predicted.indices[:, i]
        if correct is None:
            correct = pred_picks == actual_picked
        else:
            correct |= pred_picks == actual_picked
    recall = correct.sum() / correct.shape[0]
    return recall