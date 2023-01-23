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

import datetime
import os
import time

from enum import Enum

from arena_draft_bot.arena_draft_manager import ArenaDraftManager
from set_config import SetConfig

class LogDirChecker:
    def __init__(self, arena_log_dir) -> None:
        self.arena_log_dir = arena_log_dir
        self.initial_files = sorted(os.listdir(arena_log_dir))

    def check_for_log_file(self):
        files_in_dir = sorted(os.listdir(self.arena_log_dir))
        if not files_in_dir:
            return None
        if len(files_in_dir) > len(self.initial_files):
            return os.path.join(self.arena_log_dir, files_in_dir[-1])
        return None

class LogFileReader:
    """
    Simple log reader that executes the given callbacks when it matches
    particular log patterns.
    """
    def __init__(self, log_file, start_handler, pack_handler, pick_handler) -> None:
        """
        INPUTS:
            - log_file: Path to log.
            - start_handler: Function to execute on start line.
            - pack_handler: Function to execute on pack line.
            - pick_handler: Function to execute on pick line.
        
        The caller must ensure that these functions will still be valid when called
        (the functions should not depend on classes / data that may be deleted later on
        in the program).
        """
        self.log_file = log_file
        self.current_line = 0
        self.matcher_handler_map = {
            self._is_start_line : start_handler,
            self._is_draft_pack : pack_handler,
            self._is_draft_pick : pick_handler
        }

    def _is_start_line(self, line):
        return "Client.SceneChange" in line and "HumanDraft" in line
    
    def _is_draft_pack(self, line):
        return "Draft.Notify" in line and "PackCards" in line

    def _is_draft_pick(self, line):
        return "PlayerDraftMakePick" in line and "GrpId" in line

    def read_logs_until_match_and_handle(self):
        with open(self.log_file, "r") as f:
            for i, line in enumerate(f):
                if i < self.current_line:
                    continue
                self.current_line = i
                for matcher in self.matcher_handler_map:
                    if matcher(line):
                        self.current_line += 1
                        self.matcher_handler_map[matcher](line)
                        return

class BotState(Enum):
    # Arena hasn't been started. Waiting on logs to read.
    WAITING_FOR_LOG = 1
    # Arena has been started, and the log file is being read. No draft is
    # currently active.
    ATTACHED_TO_LOG = 2
    # The player is in a draft.
    DRAFTING = 3

class ArenaDraftBot:
    """
    Class to handle running the draft bot against logs of an arena
    draft.
    """
    def __init__(self, per_set_config: SetConfig, arena_log_dir, model_file, existing_log_file=None) -> None:
        """
        INPUT:
            - per_set_config: Info about the set being drafted (see set_config.py).
            - arena_log_dir: Dir containing arena logs. See
              https://mtgarena-support.wizards.com/hc/en-us/articles/360000726823-Creating-Log-Files-on-PC-Mac
            - model_file: Path to pickled model for this set.
            - existing_log_file: Log file to attach to right away (optional).
              If not provided, the bot will wait for a new log file to appear
              in the directory.
        """
        self.per_set_config = per_set_config
        self.arena_log_dir = arena_log_dir
        self.model_file = model_file
        if existing_log_file is not None:
            self.bot_state = BotState.ATTACHED_TO_LOG
            self._set_up_logfilereader(os.path.join(arena_log_dir, existing_log_file))
        else:
            self.bot_state = BotState.WAITING_FOR_LOG
            self.log_dir_checker = LogDirChecker(arena_log_dir)
            self.log_file_reader = None
        self.draft_manager = None
    
    def _handle_start_line(self, line):
        print("Found draft start line, starting draft mode.")
        self.bot_state = BotState.DRAFTING
        draft_uid = "arena_draft_{0}".format(datetime.datetime.now())
        # Note that this overrides the existing manager if a new draft starts.
        self.draft_manager = ArenaDraftManager(self.per_set_config, self.model_file, draft_uid)

    def _handle_pack_line(self, line):
        self.draft_manager.handle_pack_line(line)

    def _handle_pick_line(self, line):
        self.draft_manager.handle_pick_line(line)

    def _set_up_logfilereader(self, log_file_path):
        self.log_file_reader = LogFileReader(
            log_file_path, 
            start_handler=lambda x: self._handle_start_line(x), 
            pack_handler=lambda x: self._handle_pack_line(x), 
            pick_handler=lambda x: self._handle_pick_line(x)
        )

    def _check_for_log_file(self):
        log_file = self.log_dir_checker.check_for_log_file()
        if log_file is not None:
            print("Found log file ", log_file)
            self.bot_state = BotState.ATTACHED_TO_LOG
            self._set_up_logfilereader(log_file)

    def run(self):
        """
        Run the main processing loop.
        """
        while True:
            if self.bot_state == BotState.WAITING_FOR_LOG:
                self._check_for_log_file()
            elif self.bot_state == BotState.ATTACHED_TO_LOG:
                self.log_file_reader.read_logs_until_match_and_handle()
            elif self.bot_state == BotState.DRAFTING:
                self.log_file_reader.read_logs_until_match_and_handle()

            # Limit the bot to 10 actions per second.
            time.sleep(0.1)
