import os
import random
import torch
import re

from arena_draft_bot import pick_display_utils
from interp.autointerp_batch_prompt_runner import AutointerpResult
from set_config import SetConfig

def _mask_rows_by_sum_past_n(tensor, N, target_sum):
    if N > tensor.shape[1]:
        raise ValueError("N cannot be greater than the number of columns in the tensor.")

    past_n_elements = tensor[:, N:]

    # Calculate the sum of the elements past the Nth for each row
    sum_past_n = torch.sum(past_n_elements, dim=1)

    # Create a boolean mask where the sum is equal to the target_sum
    mask = (sum_past_n == target_sum)
    return mask


class OutputBasedAutoInterpQuizEvaluator:
    """
    A class for evaluating the output-based autointerp quiz.

    The quiz is a multiple choice quiz where the LLM is given a pool of cards and two sets of neurons.
    The LLM must pick the set of neurons that is more likely to be the correct set of neurons that activated in the example.

    The autointerp_results_dir must contain the autointerp results for the given latent indices (see
    output_based_autointerp.py for more details).
    """
    def __init__(self, set_config: SetConfig, autointerp_results_dir: str):
        self.pick_displayer = pick_display_utils.PickDisplayer(set_config)
        self.set_config = set_config
        self.autointerp_results_dir = autointerp_results_dir
        assert os.path.exists(self.autointerp_results_dir), f"Autointerp results directory {self.autointerp_results_dir} does not exist."


    def get_sample_indices_to_evaluate(self, inputs, pick_num_to_focus):
        """
        Get a set of sample indices to evaluate.

        Inputs:
            inputs: A tensor of shape (N, D) where N is the number of samples and D is the number of features.
            pick_num_to_focus: Return all samples where the pool contains this number of cards (ie we are on that pick number).

        Returns:
            A 1D tensor of where each element is the index of a sample that should be evaluated.
        """
        return torch.nonzero(_mask_rows_by_sum_past_n(inputs, self.set_config.set_size, pick_num_to_focus))


    def extract_label_from_autointerp_files(self, latent_idxs):
        """
        Extract the label from the autointerp results files for the given latent indices.

        If a file is missing or the label is not found, the latent index will be mapped to None.
        """
        extracted_labels = {}
        for latent_idx in latent_idxs:
            file_path = os.path.join(self.autointerp_results_dir, f"latent_{latent_idx}.txt")
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    # Extract text between <label> and </label>
                    label_match = re.findall(r'<label>(.*?)</label>', file_content, re.DOTALL)
                    if label_match:
                        extracted_labels[latent_idx] = label_match[1]
                    else:
                        extracted_labels[latent_idx] = "Label not found"
            except FileNotFoundError:
                extracted_labels[latent_idx] = None
            except Exception as e:
                extracted_labels[latent_idx] = None
        return extracted_labels


    def build_quiz_prompt_for_sample_index(self, example_index, latents, inputs, topk=10):
        """
        Build a quiz prompt for a given sample index.

        The prompt will show the input pool and two sets of neurons: A random set of neurons that didn't activate in the example,
        and a set of neurons that highly activated in the example.

        The prompt will ask the LLM to pick the set of neurons that is more likely to be the correct set of neurons that activated
        in the example.

        Note that self.autointerp_results_dir must contain the autointerp results for the given latent indices (see
        output_based_autointerp.py for more details).

        Returns:
            str: The prompt text, or None if an error occurred.
            int: The correct answer, or None if an error occurred.
        """
        top_latents = torch.topk(latents, k = topk, dim = 1)

        top_latent_idxs = top_latents.indices[example_index].tolist()

        # Find latent indices that are 0 in the target example row
        zero_latent_indices = [i for i, value in enumerate(latents[example_index]) if value == 0]

        # Pick random zero latent indices
        num_indices_to_pick = topk
        if len(zero_latent_indices) >= num_indices_to_pick:
            random_zero_latent_indices = random.sample(zero_latent_indices, num_indices_to_pick)
        else:
            print(f"Only found {len(zero_latent_indices)} latent indices with zero activation in example {example_index}. Picking all of them.")
            random_zero_latent_indices = zero_latent_indices
        
        top_latents_labels = self.extract_label_from_autointerp_files(top_latent_idxs)
        random_latents_labels = self.extract_label_from_autointerp_files(random_zero_latent_indices)
        
        # Check if all values in both label dicts are None. If so, there's nothing useful here.
        if all(label is None for label in top_latents_labels.values()) and all(label is None for label in random_latents_labels.values()):
            return None, None

        # Neuron label information.
        top_latents_str = "\n".join([f"Latent Index {idx}: {label}" for idx, label in top_latents_labels.items()])
        random_latents_str = "\n".join([f"Latent Index {idx}: {label}" for idx, label in random_latents_labels.items()])

        # Pick 1 or 2 randomly.
        correct_answer_random_choice = random.choice([1, 2])
        first_neurons = top_latents_str if correct_answer_random_choice == 1 else random_latents_str
        second_neurons = random_latents_str if correct_answer_random_choice == 1 else top_latents_str

        # Pool information.
        pool_names = self.pick_displayer._card_vector_to_card_names(inputs[example_index][self.set_config.set_size:])
        pool_names = pick_display_utils.expand_card_counts(pool_names)
        pool_name_strings = []
        for name in pool_names:
            card_metadata = self.pick_displayer.local_cache_of_card_json[name]
            pool_name_strings.append(f"{name},{card_metadata['color_identity']},{card_metadata['type_line']},{card_metadata['mana_cost']},{card_metadata['rarity']}")
        pool_names_for_prompt = "\n".join(pool_name_strings)


        example_prompt = f"""
Pool:
{pool_names_for_prompt}

Potential set of neurons 1:
{first_neurons}

Potential set of neurons 2:
{second_neurons}
        """

        return example_prompt, correct_answer_random_choice
