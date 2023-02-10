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
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

import set_config
from training.iterable_draft_dataset import IterableDraftDataset
from training.draft_pick_lstm import DraftPickLSTM
from training.draft_dataset_splitter import DraftDatasetSplitter

# Per-set configuration parameters.
_SET_CONFIG = None

def train(lstm, trainloader, optimizer, criterion, epochs, cuda_device):
    """
    Optimizes `criterion` using the algorithm specified by `optimizer` to
    train `lstm` on the `trainloader` training set for `epochs` iterations.

    This is a pretty standard traning loop emulating the intro instructions at
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    if cuda_device is not None:
        lstm.to(cuda_device)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a packed tensor of
            # [pack, pool, pick]. Split into inputs and labels.
            data = data.to(torch.float32)
            # Swap first and second dims, since LSTM expects first dim
            # to be sequence, second to be batch.
            data = data.transpose(0, 1)
            if cuda_device is not None:
                data = data.to(cuda_device)
            inputs = data[:, :, :(_SET_CONFIG.set_size * 2)]
            labels = data[:, :, (_SET_CONFIG.set_size * 2):]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = lstm(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print("Finished Training.")


def train_and_save_lstm(data_dir, out_dir, train_on_gpu):
    """
    Trains on the split dataset contained in `data_dir`. Saves the resulting
    model to `out_dir`.

    Model file name will be timestamped with the time it was trained.
    """
    cuda_device = None
    if train_on_gpu:
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")
            print("Starting training on device: ", cuda_device)
        else:
            print("GPU training not available, continuing on CPU.")
    lstm = DraftPickLSTM(card_pool_size=_SET_CONFIG.set_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(),  lr=0.001, betas=(0.9, 0.999))
    dataset_splitter = DraftDatasetSplitter(data_dir)
    trainloader = torch.utils.data.DataLoader(
        IterableDraftDataset(dataset_splitter.get_training_splits()), 
        batch_size=1000
    )
    train(lstm, trainloader, optimizer, criterion, epochs=40, cuda_device=cuda_device)
    # Move lstm to cpu if it was trained on gpu.
    lstm.to(torch.device("cpu"))
    model_file = "draft_bot_lstm_{0}.pt".format(datetime.datetime.now())
    print("Saving trained model to ", model_file)
    torch.save(lstm, os.path.join(out_dir, model_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train tan lstm on draft data.')
    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        help='Directory to load split dataset from. Must be sequential dataset.')
    parser.add_argument('-o','--out-dir', type=str, dest='out_dir',
                        help='Directory to write the trained model to.')
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
    train_and_save_lstm(data_dir=args.data_dir, out_dir=args.out_dir, train_on_gpu=args.gpu)
