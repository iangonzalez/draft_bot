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
from training.draft_pick_nnet import DraftPickNN
from training.draft_dataset_splitter import DraftDatasetSplitter

# Per-set configuration parameters.
_SET_CONFIG = None

def train(net, trainloader, optimizer, criterion, epochs):
    """
    Optimizes `criterion` using the algorithm specified by `optimizer` to
    train `net` on the `trainloader` training set for `epochs` iterations.

    This is a pretty standard traning loop emulating the intro instructions at
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a packed tensor of
            # [pack, pool, pick]. Split into inputs and labels.
            data = data.to(torch.float32)
            inputs = data[:, :(_SET_CONFIG.set_size * 2)]
            labels = data[:, (_SET_CONFIG.set_size * 2):]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print("Finished Training.")


def train_and_save_nnet(data_dir, out_dir):
    """
    Trains on the split dataset contained in `data_dir`. Saves the resulting
    model to `out_dir`.

    Model file name will be timestamped with the time it was trained.

    TODO(iangonzalez): Add cuda training capability to this function.
    """
    net = DraftPickNN(card_pool_size=_SET_CONFIG.set_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),  lr=0.001, betas=(0.9, 0.999))
    dataset_splitter = DraftDatasetSplitter(data_dir)
    trainloader = torch.utils.data.DataLoader(
        IterableDraftDataset(dataset_splitter.get_training_splits()), 
        batch_size=100
    )
    train(net, trainloader, optimizer, criterion, epochs=20)
    torch.save(net, os.path.join(out_dir, "draft_bot_nnet_{0}.pt".format(datetime.datetime.now())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the nnet on draft data.')
    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        help='Directory to load split dataset from.')
    parser.add_argument('-o','--out-dir', type=str, dest='out_dir',
                        help='Directory to write the trained model to.')
    parser.add_argument('-s', '--set-id', type=str, default='BRO', dest='set_id',
                        help='set id of the drafts being tested (configuration purposes).')
    parser.add_argument('--set-config-path', type=str, required=False, dest='set_config_path',
                        help='set config of the drafts being tested (configuration purposes). Overrides set-id.')
    args = parser.parse_args()
    if args.set_config_path is not None:
        _SET_CONFIG = set_config.get_set_config_from_path(args.set_config_path)
    else:
        _SET_CONFIG = set_config.get_set_config(args.set_id)
    train_and_save_nnet(data_dir=args.data_dir, out_dir=args.out_dir)
