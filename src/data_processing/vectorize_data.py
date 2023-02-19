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
import os

import pandas as pd
import numpy as np
import torch

# Number of rows to be processed at a time. Configurable depending on available
# RAM on the worker machine.
_ROW_BATCH_SIZE = 500000

# Number of picks assumed to be in each draft.
_PICK_COUNT_DIM = 45

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


def get_valid_draft_ids(df, pick_count_dim):
    id_counts = pd.value_counts(df["draft_id"])
    return [id for id in id_counts.keys() if id_counts[id] == pick_count_dim]


def filter_for_valid_drafts(df):
    valid_draft_ids = set(get_valid_draft_ids(df, _PICK_COUNT_DIM))
    valid_draft_idxs = df["draft_id"].apply(lambda x: x in valid_draft_ids)
    return df.loc[valid_draft_idxs]


def add_pick_num_dimension(draft_tensor_2d):
    assert draft_tensor_2d.shape[0] % _PICK_COUNT_DIM == 0
    draft_count_dim = int(draft_tensor_2d.shape[0] / _PICK_COUNT_DIM)
    feature_count_dim = draft_tensor_2d.shape[1]
    return draft_tensor_2d.reshape((draft_count_dim, _PICK_COUNT_DIM, feature_count_dim))


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


def filter_drafts_for_high_win_rate(df):
    win_rate_bucket = df['user_game_win_rate_bucket']
    avg_win_rate = win_rate_bucket.mean()
    return df.loc[win_rate_bucket > avg_win_rate]


def vectorize_and_store_dataset(data_file, out_dir, is_sequential, filter_by_win_rate):
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

        if filter_by_win_rate:
            df = filter_drafts_for_high_win_rate(df)
        if is_sequential:
            # In the sequential case, every draft must be a sequence of picks
            # of the expected size (_PICK_COUNT_DIM) or the reshape will fail.
            df = filter_for_valid_drafts(df)

        # Create vectorized representation of picks.
        pick_matrix = np.zeros((df.shape[0], len(card_index_map)), dtype=np.int8)
        for i, pick in enumerate(df["pick"]):
            card_names_to_vector([pick], card_index_map, pick_matrix[i])
        
        # Data frame already contains representation of pack and pool, just get the columns
        # that contain that data.
        canonicalized_pack_and_pool_vectors_df = df[sorted(pack_cols) + sorted(pool_cols)]

        # Concatentate the two.
        draft_tensor = torch.cat((torch.tensor(canonicalized_pack_and_pool_vectors_df.values), 
                                 torch.tensor(pick_matrix)), dim=1)
        if is_sequential:
            # Reshape to add a dimension for per-draft pick number.
            draft_tensor = add_pick_num_dimension(draft_tensor)

        out_file_path = os.path.join(out_dir, output_file_pattern.format(chunk_number))
        print("Saving a vectorized data matrix of size", draft_tensor.shape, "at",  out_file_path)
        torch.save(draft_tensor, out_file_path)
        chunk_number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a raw 17lands csv file into a vectorized format usable for training.')
    parser.add_argument('-d', '--data-file', type=str, dest='data_file',
                        help='.csv file to load the dataset from.')
    parser.add_argument('-o','--out-dir', type=str, dest='out_dir',
                        help='Directory to write the vectorized data to.')
    parser.add_argument('--sequential', action='store_true',
                        help='Toggle to store the data as a sequence of picks (grouped by draft id).')
    parser.add_argument('--win-rate-filter', action='store_true', dest='win_rate_filter',
                        help='Toggle to filter draft picks by player win rate (upper half of distrubution).')
    args = parser.parse_args()
    vectorize_and_store_dataset(args.data_file, args.out_dir, args.sequential,
                                args.win_rate_filter)
