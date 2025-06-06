import argparse
import asyncio
from collections import Counter

import torch

import set_config
from interp.autointerp_batch_prompt_runner import (
    create_unfilled_autointerp_results,
    fill_autointerp_results,
    save_latent_autointerp_results,
    save_quiz_autointerp_results
)
from interp.claude_autointerp_client import AutoInterpMode, ClaudeAutoInterpClient
from interp.output_based_autointerp import OutputBasedAutoInterp
from interp.output_based_autointerp_quiz_evaluator import OutputBasedAutoInterpQuizEvaluator
from training.draft_dataset_splitter import DraftDatasetSplitter
from training.iterable_draft_dataset import IterableDraftDataset
from training.sae import SparseAutoencoder
from training.sae_instrumented_draft_pick_nnet import SAEInstrumentedDraftPickNN

_SET_CONFIG = None


def sample_test_batch(data_dir, draft_net, sample_size, sae_instrumented_layer_idx=3):
    """Taking a closer look at one test batch. Get all relevant information for N picks."""
    data = None
    inputs = None
    labels = None
    y = None
    latents = None
    pre_latent_hidden_activations = None
    draft_net.eval()
    dataset_splitter = DraftDatasetSplitter(data_dir)
    testloader = torch.utils.data.DataLoader(
        IterableDraftDataset(dataset_splitter.get_validation_splits()),
        batch_size=sample_size
    )
    with torch.no_grad():
        for data in testloader:
            data = data.to(torch.float32)
            data = data.to("cuda")
            inputs = data[:, :(_SET_CONFIG.set_size * 2)]
            labels = data[:, (_SET_CONFIG.set_size * 2):]

            y = draft_net(inputs)
            if sae_instrumented_layer_idx is not None:
                activations_dict = draft_net.get_most_recent_latent_activations()
                layer_key = f"layer{sae_instrumented_layer_idx}"
                if layer_key in activations_dict:
                    latents = activations_dict[layer_key]
                else:
                    print(f"Layer {layer_key} not found in activations_dict")
            pre_latent_hidden_activations = draft_net.get_most_recent_hidden_activations()[f"layer{sae_instrumented_layer_idx}"]
            break  # stop after one iteration

    actual_picked = torch.argmax(labels, 1)
    return inputs, y, actual_picked, latents, pre_latent_hidden_activations


async def run_autointerp_for_sae(data_dir, output_directory, sae_file, draft_net_file, sample_size, sae_layer_idx, mode):
    # Load the SAE and DraftPickNN.
    sae = torch.load(sae_file, weights_only=False)
    draft_net = torch.load(draft_net_file, weights_only=False)

    sae_instrumented_draft_net = SAEInstrumentedDraftPickNN(
        draft_net,
        sae_model=sae,
        sae_replaces_dropout_output_idx=sae_layer_idx
    )

    print("Loading validation dataset...")
    inputs, y, _, latents, _ = sample_test_batch(
        data_dir,
        sae_instrumented_draft_net,
        sample_size,
        sae_layer_idx
    )

    if mode == "AUTOINTERP_ON_OUTPUTS":
        output_based_auto_interp = OutputBasedAutoInterp(set_config=_SET_CONFIG, sae_instrumented_draft_net=sae_instrumented_draft_net)
        print("Generating prompts for autointerp on outputs...")
        prompts = []
        for latent_idx in range(latents.shape[1]):
            prompt = output_based_auto_interp.get_activation_diff_prompt_text_for_latent(latent_idx, inputs)
            prompts.append(prompt)
        auto_interp_results = create_unfilled_autointerp_results(prompts)
        
        print(f"Sending {len(prompts)} autointerp on outputs prompts to Claude...")
        claude_client = ClaudeAutoInterpClient(_SET_CONFIG, auto_interp_mode=AutoInterpMode.AUTOINTERP_ON_OUTPUTS)
        await fill_autointerp_results(auto_interp_results, claude_client)
        save_latent_autointerp_results(auto_interp_results, output_directory)
  
    elif mode == "AUTOINTERP_EVAL_OUTPUT_BASED_FEATURES":
        output_based_auto_interp_quiz_evaluator = OutputBasedAutoInterpQuizEvaluator(set_config=_SET_CONFIG, autointerp_results_dir=output_directory)
        print("Generating prompts for autointerp on outputs quiz...")
        prompts = []
        answers = []
        for sample_idx in output_based_auto_interp_quiz_evaluator.get_sample_indices_to_evaluate(inputs, pick_num_to_focus=25):
            sample_idx = sample_idx.item()
            prompt, answer = output_based_auto_interp_quiz_evaluator.build_quiz_prompt_for_sample_index(sample_idx, latents, inputs)
            prompts.append(prompt)
            answers.append(answer)
        auto_interp_results = create_unfilled_autointerp_results(prompts)

        print(f"Sending {len(prompts)} autointerp on outputs quiz prompts to Claude...")
        claude_client = ClaudeAutoInterpClient(_SET_CONFIG, auto_interp_mode=AutoInterpMode.AUTOINTERP_EVAL_OUTPUT_BASED_FEATURES)
        await fill_autointerp_results(auto_interp_results, claude_client)
        save_quiz_autointerp_results(auto_interp_results, answers, output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run AutoInterp pipeline for a target SAE / DraftPickNN pair.'
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        dest='data_dir',
        help='Directory to load validation dataset from. Autointerp will sample from this dataset.'
    )
    parser.add_argument(
        '-o', '--output-directory',
        type=str,
        default=None,
        dest='output_directory',
        help='Directory to save the results to.'
    )
    parser.add_argument(
        '-s', '--sae-file',
        type=str,
        default=None,
        dest='sae_file',
        help='File name of the SAE to use.'
    )
    parser.add_argument(
        '--draft-net-file',
        type=str,
        default=None,
        dest='draft_net_file',
        help='File name of the DraftPickNN to use.'
    )
    parser.add_argument(
        '--set-id',
        type=str,
        default='BRO',
        dest='set_id',
        help='set id of the drafts being tested (configuration purposes).'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        dest='sample_size',
        help='Number of picks to sample from the validation dataset.'
    )
    parser.add_argument(
        '--sae-layer-idx',
        type=int,
        default=3,
        dest='sae_layer_idx',
        help='Index of the layer whose output will be replaced by the SAE\'s encoded features.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='autointerp_on_outputs',
        dest='mode',
        help='Mode to run the autointerp in. Should exactly match an option in claude_autointerp_client.AutoInterpMode.'
    )
    args = parser.parse_args()
    _SET_CONFIG = set_config.get_set_config(args.set_id)
    asyncio.run(run_autointerp_for_sae(
        args.data_dir,
        args.output_directory,
        args.sae_file,
        args.draft_net_file,
        args.sample_size,
        args.sae_layer_idx,
        args.mode
    ))