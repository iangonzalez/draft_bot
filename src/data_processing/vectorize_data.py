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

import os
import sys

import pandas as pd
import numpy as np
import torch

# Number of rows to be processed at a time. Configurable depending on available
# RAM on the worker machine.
_ROW_BATCH_SIZE = 500000

def is_file_valid(data_file):
    """
    INPUT:
      - data_file: File path to validate.
    
    OUTPUT:
        Returns False if this is not a .csv extension file, or if it is not a
    valid file at all.
    """
    if not data_file.endswith(".csv"):
        print("Not a .csv")
        return False
    if not os.path.isfile(data_file):
        print("Not a file.")
        return False
    return True


def count_rows_csv(data_file):
    return sum(1 for line in open(data_file))


def get_pack_and_pool_columns(data_file):
    """
    INPUT:
        - data_file: File path of draft data csv to read.
    
    OUTPUT:
        Tuple of:
        - pack_cols: List of the pack-related columns in the csv.
        - pool_cols: List of the pool-related columns in the csv.

        Note that each column will correspond to a card name.
    """
    tiny_df =  pd.read_csv(data_file, nrows=1)
    pool_cols = sorted([col for col in tiny_df.columns if "pool_" in col])
    pack_cols = sorted([col for col in tiny_df.columns if "pack_card" in col])
    return pack_cols, pool_cols


def get_sorted_card_names(pack_cols):
    prefix = "pack_card_"
    return [s[len(prefix):] if s.startswith(prefix) else s for s in pack_cols]


def card_names_to_vector(names, card_index_map, card_vec):
  """
  INPUT:
  - Iterable of card names. Invalid names will cause a crash.
  - card_index_map: Mapping of names to vectorized index.
  - Array of size N (where N is card name list length). Expects that the array
    is filled with ints (ie created via np.zeros()).
  """
  for name in names:
    assert name in card_index_map.keys()
    card_vec[card_index_map[name]] += 1


def vectorize_and_store_dataset(data_file, out_dir):
    """
    Vectorizes the draft data and saves as a series of pickled tensors.

    Each row of the tensor will be of size N * 3, where N is the set size. The
    first N elements are a vectorized representation of the pack, the second N
    are the pool, and the last N are the pick that the human made.

    The tensors will be saved in a series of splits (split0...splitX) where the
    number of required splits is determined by the size of input and the global
    param _ROW_BATCH_SIZE.

    INPUT: 
        - data_file: Takes a file path pointing to a draft data .csv file downloaded
        from 17lands.
        - out_dir: Directory to put the output data in.
    """
    assert is_file_valid(data_file)

    row_count = count_rows_csv(data_file)
    print("Vectorizing ", row_count, " rows in batches of ", _ROW_BATCH_SIZE)

    pack_cols, pool_cols = get_pack_and_pool_columns(data_file)
    card_index_map = { s:i for i,s in enumerate(get_sorted_card_names(pack_cols)) }

    # Leading 0s for alphabetizing purposes.
    output_file_pattern = os.path.basename(data_file) + ".vectorized.split{:03d}.pkl"

    type_map = { s: np.int8 for s in pool_cols + pack_cols }
    df_iterator = pd.read_csv(data_file, dtype=type_map, chunksize=_ROW_BATCH_SIZE)

    chunk_number = 0
    for chunk in df_iterator:
        df = pd.DataFrame(chunk)
        # Create representation of picks.
        pick_matrix = np.zeros((df.shape[0], len(card_index_map)), dtype=np.int8)
        for i, pick in enumerate(df["pick"]):
            card_names_to_vector([pick], card_index_map, pick_matrix[i])
        
        # Data frame already contains representation of pack and pool, just get the columns
        # that contain that data.
        canonicalized_pack_and_pool_vectors_df = df[sorted(pack_cols) + sorted(pool_cols)]

        # Concatentate the two, then save.
        processed_draft_tensor = torch.cat((torch.tensor(canonicalized_pack_and_pool_vectors_df.values), 
                                 torch.tensor(pick_matrix)), dim=1)
        out_file_path = os.path.join(out_dir, output_file_pattern.format(chunk_number))
        print("Saving a vectorized data matrix of size", processed_draft_tensor.shape, "at",  out_file_path)
        torch.save(processed_draft_tensor, out_file_path)
        chunk_number += 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: vectorize_data.py <input csv> <output dir>")
    vectorize_and_store_dataset(sys.argv[1], sys.argv[2])
