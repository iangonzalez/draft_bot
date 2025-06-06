import torch
from dataclasses import dataclass
from arena_draft_bot import pick_display_utils
from training.sae_instrumented_draft_pick_nnet import SAEInstrumentedDraftPickNN
from set_config import SetConfig


@dataclass
class ActivationDiffExample:
  latent_idx: int
  top_positive_card_names: list[str]
  top_positive_scores: list[float]
  top_negative_card_names: list[str]
  top_negative_scores: list[float]


class OutputBasedAutoInterp:
    """
    A class for output-based autointerp.

    For a given example that activates the target latent:
     - Get the "pre-pack" activation layer (the model's scoring of all the cards)
     - Get the "pre-pack" activation again, this time with the latent amplified
     - Diff the two to see what changed

    Then create an interpretability prompt for an LLM: 
    Which kinds of cards are strongly promoted or strongly demoted by this feature?
    """
    def __init__(self, set_config: SetConfig, sae_instrumented_draft_net: SAEInstrumentedDraftPickNN):
        self.pick_displayer = pick_display_utils.PickDisplayer(set_config)
        self.sae_instrumented_draft_net = sae_instrumented_draft_net

    def get_prepack_activation_diff(self, latent_idx, multiplier, inputs):
        """
        Modifies the SAE to multiply a specific latent activation, and calculates the diff
        in prepack activations between the modified and unmodified SAE.

        Args:
            latent_idx: The index of the latent feature to modify.
            multiplier: The factor to multiply the latent activation by.
            sae_instrumented_draft_net: The SAEInstrumentedDraftPickNN instance.
            inputs: The input tensor to the network.

        Returns:
            A PyTorch tensor representing the difference in prepack activations.
        """
        self.sae_instrumented_draft_net.eval()
        # Clear any existing multipliers.
        self.sae_instrumented_draft_net.clear_latent_multipliers()
        # Get prepack activations with the original SAE
        with torch.no_grad():
            _ = self.sae_instrumented_draft_net(inputs)
            original_prepack_activations = self.sae_instrumented_draft_net.get_most_recent_prepack_activations()

        # Set the new multipliers in the SAE
        self.sae_instrumented_draft_net.set_latent_multipliers([latent_idx], [multiplier])

        # Get prepack activations with the modified SAE
        with torch.no_grad():
            _ = self.sae_instrumented_draft_net(inputs)
            modified_prepack_activations = self.sae_instrumented_draft_net.get_most_recent_prepack_activations()

        # Calculate the difference
        diff = modified_prepack_activations - original_prepack_activations
        # Reset the draft net back to normal mode before returning.
        self.sae_instrumented_draft_net.clear_latent_multipliers()

        return diff

    def get_activation_diff_examples_for_latent(self, latent_idx, inputs):
        """
        Get the top positive and negative card names / scores in the prepack activation diff for a given latent index.
        """
        diff = self.get_prepack_activation_diff(latent_idx=latent_idx, multiplier=10, inputs=inputs)
        nonzero_mask = (diff.sum(dim=1) != 0)
        if nonzero_mask.sum() == 0:
            print(f"Found no nonzero diffs for latent idx {latent_idx}")
            return None
        top_positive_changes = torch.topk(diff, k = 30, dim = 1)
        top_negative_changes = torch.topk(diff * -1, k = 30, dim = 1)
        # Arbitrarily pick the first nonzero example. They all seem to be well correlated.
        # TODO: Pick the best example. This may not hold true in the future.
        top_positive_names = [self.pick_displayer.sorted_card_names[idx] for idx in top_positive_changes.indices[nonzero_mask][0].tolist()]
        top_positive_scores = top_positive_changes.values[nonzero_mask][0].tolist()
        top_negative_names = [self.pick_displayer.sorted_card_names[idx] for idx in top_negative_changes.indices[nonzero_mask][0].tolist()]
        top_negative_scores = (top_negative_changes.values[nonzero_mask][0] * -1).tolist()
        return ActivationDiffExample(
            latent_idx=latent_idx,
            top_positive_card_names=top_positive_names,
            top_positive_scores=top_positive_scores,
            top_negative_card_names=top_negative_names,
            top_negative_scores=top_negative_scores
        )

    def get_activation_diff_prompt_text_for_latent(self, latent_idx, inputs):
        """
        Get the output-based autointerp prompt text for a given latent index.

        The prompt will show the top positive and negative cards in the prepack activation diff for the latent index
        and ask the LLM to label the latent index as a feature.
        """
        diff_example = self.get_activation_diff_examples_for_latent(latent_idx, inputs)
        
        if diff_example is None:
            return None
        positive_card_strings = []
        for name, score in zip(diff_example.top_positive_card_names, diff_example.top_positive_scores):
            card_metadata = self.pick_displayer.local_cache_of_card_json[name]
            positive_card_strings.append(f"{name},{card_metadata['color_identity']},{card_metadata['type_line']},{card_metadata['mana_cost']},{card_metadata['rarity']}  Score: {score}")
        positive_card_block = "\n".join(positive_card_strings)
        negative_card_strings = []
        for name, score in zip(diff_example.top_negative_card_names, diff_example.top_negative_scores):
            card_metadata = self.pick_displayer.local_cache_of_card_json[name]
            negative_card_strings.append(f"{name},{card_metadata['color_identity']},{card_metadata['type_line']},{card_metadata['mana_cost']},{card_metadata['rarity']}  Score: {score}")
        negative_card_block = "\n".join(negative_card_strings)
        postscript = "\nPlease provide the label for this neuron between the <label> and </label> tags." 
        return f"<example>\n{positive_card_block}\n{negative_card_block}\n</example>" + postscript