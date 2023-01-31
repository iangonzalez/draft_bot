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

# Functions and classes for displaying draft information and bot predictions
# to the user.
import json
import os

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from set_config import SetConfig


def draw_red_outline_around_image(image):
    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Get the width and height of the image
    width, height = image.size

    # Draw a red outline around the image
    draw.rectangle((0, 0, width, height), outline=(255, 0, 0), width=10)


def expand_card_counts(names_with_counts):
    names = []
    for name, count in names_with_counts:
        names.extend([name for i in range(int(count.item()))])
    return names


class PickDisplayer:
    """
    Displays pick / prediction information to the console and to a figure using
    pyplot.
    """
    def __init__(self, per_set_config: SetConfig, draft_uid) -> None:
       self.per_set_config = per_set_config
       self.sorted_card_names = json.load(open(per_set_config.card_list_path, "r"))
       self.card_name_to_index = {c: i for i, c in enumerate(self.sorted_card_names)}
       self.image_dir = per_set_config.card_image_dir
       self.draft_uid = draft_uid


    def _get_scores_for_cards(self, names, predictions):
        return [predictions[self.card_name_to_index[n]].item() for n in names]


    def _card_vector_to_card_names(self, card_vec):
        name_list = []
        for i, value in enumerate(card_vec):
            if value != 0:
                name_list.append((self.sorted_card_names[i], value))
        return name_list
    

    def _top_n_from_preds(self, predictions, n):
        assert predictions.shape == (1, self.per_set_config.set_size)
        ranked_card_names = self._card_vector_to_card_names(predictions[0])
        
        # Pull out the float from the dim 0 tensors for better printing.
        ranked_card_names = [(name_value[0], name_value[1].item()) for name_value in ranked_card_names]

        ranked_card_names.sort(key = lambda x: x[1])
        ranked_card_names.reverse()
        if n <= len(ranked_card_names):
            return ranked_card_names[:n]
        return ranked_card_names


    def _plot_cards_by_name(self, names, pack_num=0, pick_num=0, per_image_scores=None):
        index_max = None
        if per_image_scores:
            assert len(per_image_scores) == len(names)
            index_max = max(range(len(per_image_scores)),
                            key=per_image_scores.__getitem__)

        num_images = len(names)
        # 5 images per row
        columns = 5
        rows = int((num_images - (num_images % 5)) / 5 + 1)

        # create figure
        fig = plt.figure(figsize=(columns*2, rows*2))

        for i, name in enumerate(names):
            image = Image.open(self.image_dir + "/{0}.jpg".format(name))
            if index_max is not None and index_max == i:
                draw_red_outline_around_image(image)
            fig.add_subplot(rows, columns, i+1)
            # showing image
            plt.imshow(image)
            plt.axis('off')
            if per_image_scores:
                plt.title("{:1f}".format(per_image_scores[i]))
        plt.savefig(os.path.join("/tmp", self.draft_uid + "p" + str(pack_num) + "p" + str(pick_num) + "test.png"))

    def print_current_pool_to_console(self, pool_vector):
        pool_names = self._card_vector_to_card_names(pool_vector)
        pool_names = expand_card_counts(pool_names)
        print("Current pool: ", pool_names)
    
    def print_pack_and_predictions_to_console(self, pack_vector, preds):
        pack_names = self._card_vector_to_card_names(pack_vector)
        pack_names = expand_card_counts(pack_names)
        print("Pack: ", pack_names)
        print("Model top picks for this pack: ", self._top_n_from_preds(preds, n=3), "\n\n")
    
    def plot_predictions(self, pack_vector, preds_vector):
        pack_names = self._card_vector_to_card_names(pack_vector[:self.per_set_config.set_size])
        pack_names = expand_card_counts(pack_names)
        self._plot_cards_by_name(pack_names, per_image_scores=self._get_scores_for_cards(pack_names, preds_vector))
