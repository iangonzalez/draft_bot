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

import set_config
from arena_draft_bot.arena_draft_bot import ArenaDraftBot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the draft bot on Magic Arena logs.')
    parser.add_argument('-l', '--log-dir', type=str, dest='arena_log_dir', required=True,
                        help='Directory to load arena logs from.')
    parser.add_argument('-e', '--existing-log-file', type=str, dest='existing_log_file', required=False,
                        help='Existing log file name to attach to in the log dir.')
    parser.add_argument('-m','--model-path', type=str, dest='model_path', required=False,
                        help='File to read the trained model from.')
    parser.add_argument('-s', '--set-id', type=str, default='BRO', dest='set_id',
                        help='set id of the drafts being tested (configuration purposes).')
    parser.add_argument('--set-config-path', type=str, required=False, dest='set_config_path',
                        help='set config of the drafts being tested (configuration purposes). Overrides set-id.')
    args = parser.parse_args()
    per_set_config = None
    if args.set_config_path is not None:
        per_set_config = set_config.get_set_config_from_path(args.set_config_path)
    else:
        per_set_config = set_config.get_set_config(args.set_id)

    model_path = args.model_path if args.model_path else per_set_config.model_path

    bot = ArenaDraftBot(per_set_config, args.arena_log_dir, model_path, args.existing_log_file)
    bot.run()
