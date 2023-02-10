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

def test_lstm(lstm, dataloader, cuda_device):
    lstm.eval()
    if cuda_device is not None:
        lstm.to(cuda_device)
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in dataloader:
            data = data.to(torch.float32)
            # Swap first and second dims, since LSTM expects first dim
            # to be sequence, second to be batch.
            data = data.transpose(0, 1)
            if cuda_device is not None:
                data = data.to(cuda_device)
            inputs = data[:, :, :(_SET_CONFIG.set_size * 2)]
            labels = data[:, :, (_SET_CONFIG.set_size * 2):]

            y = lstm(inputs)

            highest_predicted = torch.argmax(y, 2)
            actual_picked = torch.argmax(labels, 2)

            correct += int((highest_predicted == actual_picked).sum())
            total += highest_predicted.numel()
    
        accuracy = correct / total
        print("Validation accuracy:", accuracy, " Total picks:", int(total))


def load_and_test_lstm(data_dir, model_path, test_on_gpu):
    cuda_device = None
    if test_on_gpu:
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda:0')
            print("Starting testing on device: ", cuda_device)
        else:
            print("GPU not available, continuing on CPU.")
    dataset_splitter = DraftDatasetSplitter(data_dir)
    testloader = torch.utils.data.DataLoader(
        IterableDraftDataset(dataset_splitter.get_validation_splits()), 
        batch_size=1000
    )
    lstm = torch.load(model_path)
    test_lstm(lstm, testloader, cuda_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test an lstm on draft data.')
    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        help='Directory to load split dataset from.')
    parser.add_argument('-m','--model-path', type=str, dest='model_path',
                        help='Path to load the trained model from.')
    parser.add_argument('-s', '--set-id', type=str, default='BRO', dest='set_id',
                        help='set id of the drafts being tested (configuration purposes).')
    parser.add_argument('--set-config-path', type=str, required=False, dest='set_config_path',
                        help='set config of the drafts being tested (configuration purposes). Overrides set-id.')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    if args.set_config_path is not None:
        _SET_CONFIG = set_config.get_set_config_from_path(args.set_config_path)
    else:
        _SET_CONFIG = set_config.get_set_config(args.set_id)

    load_and_test_lstm(data_dir=args.data_dir, model_path=args.model_path, test_on_gpu=args.gpu)
