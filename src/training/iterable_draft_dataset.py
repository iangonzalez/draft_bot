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

class IterableDraftDataset(torch.utils.data.IterableDataset):
    """
    Class for loading vectorized draft data for torch ML functions.

    See data_processing/vectorize_data for details on how this data gets
    created. Each split is a pickled tensor of size (rows, N*3), where N is the
    size of the unique cards in the draft pool.
    """

    def __init__(self, draft_tensor_files) -> None:
        """
        INPUTS:
            - draft_tensor_files: List of files containing the draft tensors.
        """
        super().__init__()
        self.tensor_files = sorted(draft_tensor_files)
        # Must have some files to be valid.
        assert self.tensor_files
        print("Creating data loader for files: ", self.tensor_files)
    
    def __iter__(self):
        for filepath in self.tensor_files:
            draft_tensor = torch.load(filepath)
            for row in draft_tensor:
                yield row
