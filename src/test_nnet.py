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

import argparse
import torch

import set_config
from training.iterable_draft_dataset import IterableDraftDataset
from training.draft_dataset_splitter import DraftDatasetSplitter

_SET_CONFIG = None

def test_nnet(net, dataloader):
    net.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in dataloader:
            data = data.to(torch.float32)
            inputs = data[:, :(_SET_CONFIG.set_size * 2)]
            labels = data[:, (_SET_CONFIG.set_size * 2):]

            y = net(inputs)

            highest_predicted = torch.argmax(y, 1)
            actual_picked = torch.argmax(labels, 1)

            correct += int(sum(highest_predicted == actual_picked))
            total += len(highest_predicted)
    
        accuracy = correct / total
        print("Validation accuracy:", accuracy, " Total picks:", int(total))


def load_and_test_model(data_dir, model_path):
    dataset_splitter = DraftDatasetSplitter(data_dir)
    testloader = torch.utils.data.DataLoader(
        IterableDraftDataset(dataset_splitter.get_test_splits()), 
        batch_size=100
    )
    net = torch.load(model_path)
    test_nnet(net, testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the nnet on draft data.')
    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        help='Directory to load split dataset from.')
    parser.add_argument('-m','--model-path', type=str, dest='model_path',
                        help='Path to load the trained model from.')
    parser.add_argument('-s', '--set-id', type=str, default='BRO', dest='set_id',
                        help='set id of the drafts being tested (configuration purposes).')
    parser.add_argument('--set-config-path', type=str, required=False, dest='set_config_path',
                        help='set config of the drafts being tested (configuration purposes). Overrides set-id.')
    args = parser.parse_args()
    if args.set_config_path is not None:
        _SET_CONFIG = set_config.get_set_config_from_path(args.set_config_path)
    else:
        _SET_CONFIG = set_config.get_set_config(args.set_id)

    load_and_test_model(data_dir=args.data_dir, model_path=args.model_path)
