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

import glob
import os


class DraftDatasetSplitter:
    """
    Utility for splitting the data into test / validation / train sets.

    This is useful for datasets that are large and split across many files. For
    datasets that can easily fit into RAM, it's simpler to load it into memory
    and split there. 
    """

    def __init__(self, processed_data_basedir, percent_train=0.8, percent_cv=0.1):
        splits_paths = sorted(glob.glob(os.path.join(processed_data_basedir + "*")))
        splits_count = len(splits_paths)
        if splits_count <= 1:
            print("Can't split data set with only", splits_count, "splits")
            raise Exception
        dividing_idx_train = int(splits_count * percent_train - 1)
        dividing_idx_cv = int(splits_count * (percent_train + percent_cv) - 1)
        self.train_splits = splits_paths[:dividing_idx_train]
        self.validation_splits = splits_paths[dividing_idx_train:dividing_idx_cv]
        self.test_splits = splits_paths[dividing_idx_cv:]

    def get_training_splits(self):
        return self.train_splits
    
    def get_validation_splits(self):
        return self.validation_splits
    
    def get_test_splits(self):
        return self.test_splits

