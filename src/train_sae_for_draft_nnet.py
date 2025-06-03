import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

import set_config
from training.iterable_draft_dataset import IterableDraftDataset
from training.draft_dataset_splitter import DraftDatasetSplitter
from training.sae import SparseAutoencoder

def train_and_save_sae_on_nnet_activation(data_dir, out_dir, draft_net, input_dim, layer_idx=2, draft_net_version=None):
  # --- SAE Configuration ---
  # The input_dim for SAE is the number of features from the hooked layer.
  sae_input_dim = input_dim
  sae_latent_dim = sae_input_dim * 5 # Overcomplete representation (5 chosen arbitrarily)
  sae_l1_coeff = 1e-3  # L1 penalty on latent activations
  sae_sparsity_target = 0.05 # Desired average activation for KL divergence
  sae_kl_coeff = 1e-1 # Weight for KL divergence sparsity

  sae_model = SparseAutoencoder(
      input_dim=sae_input_dim,
      latent_dim=sae_latent_dim,
      l1_coeff=sae_l1_coeff,
      sparsity_target=sae_sparsity_target,
      kl_coeff=sae_kl_coeff
  ).to("cuda")

  # --- Optimizer for SAE ---
  sae_optimizer = optim.Adam(sae_model.parameters(), lr=1e-3)
  reconstruction_criterion = nn.MSELoss()

  # --- Main training loop ---
  num_epochs = 50 # Adjust as needed

  # Prepare dataset for training.
  dataset_splitter = DraftDatasetSplitter(data_dir)
  trainloader_sae = torch.utils.data.DataLoader(
      IterableDraftDataset(dataset_splitter.get_training_splits()),
      batch_size=1000
  )

  # Put draft net in eval mode since it isnt being trained.
  draft_net.eval()

  print("\nStarting SAE Training...")
  for epoch in range(num_epochs):
      sae_model.train()
      total_recon_loss = 0
      total_kl_loss = 0
      total_loss_sae = 0
      num_batches = 0

      for data in trainloader_sae:
          data = data.to("cuda") # Dataloader wraps data in a list
          # Sequential data case (3-dimensional). Unroll picks into 2d.
          if len(data.shape) == 3:
              data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
          # get the inputs; data is a packed tensor of
          # [pack, pool, pick]. Split into inputs and labels.
          data = data.to(torch.float32)
          inputs = data[:, :(_SET_CONFIG.set_size * 2)]
          labels = data[:, (_SET_CONFIG.set_size * 2):]

          # 1. Get activations from DraftPickNN
          # We modified DraftPickNN.forward to directly return the activations we want
          _ = draft_net(inputs)
          target_activations = draft_net.get_most_recent_hidden_activations()[f"layer{layer_idx+1}"]
          # Ensure target_activations are flat if necessary (should be [batch_size, features])
          if target_activations.dim() > 2:
              target_activations = target_activations.view(target_activations.size(0), -1)

          # 2. Train SAE
          sae_optimizer.zero_grad()

          reconstructed_activations, latent_representation = sae_model(target_activations)

          # Calculate losses
          recon_loss = reconstruction_criterion(reconstructed_activations, target_activations)
          kl_loss = sae_model.kl_divergence_loss(latent_representation)

          loss = recon_loss + kl_loss

          loss.backward()
          sae_optimizer.step()

          total_recon_loss += recon_loss.item()
          total_kl_loss += kl_loss.item()
          total_loss_sae += loss.item()
          num_batches += 1

      avg_recon_loss = total_recon_loss / num_batches
      avg_kl_loss = total_kl_loss / num_batches
      avg_total_loss = total_loss_sae / num_batches

      print(f"SAE Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {avg_total_loss:.6f}, "
            f"Recon Loss: {avg_recon_loss:.6f}, "
            f"KL Loss: {avg_kl_loss:.6f}")

  print("\nSAE Training Finished.")
  model_file = f"sae_on_draft_bot_nnet_activation{layer_idx+1}_{datetime.datetime.now()}.pt"
  if draft_net_version:
    model_file = f"sae_on_draft_bot_nnet_activation{layer_idx+1}_{draft_net_version}.pt"
  torch.save(sae_model, os.path.join(out_dir, model_file))
  print(f"\nSAE saved to {os.path.join(out_dir, model_file)}")
  return model_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an SAE on the activations of a DraftPickNN.')
    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        help='Directory to load split dataset from.')
    parser.add_argument('-o','--out-dir', type=str, dest='out_dir',
                        help='Directory to write the trained model to.')
    parser.add_argument('-s', '--set-id', type=str, default='BRO', dest='set_id',
                        help='set id of the drafts being tested (configuration purposes).')
    parser.add_argument('--set-config-path', type=str, required=False, dest='set_config_path',
                        help='set config of the drafts being tested (configuration purposes). Overrides set-id.')
    parser.add_argument('--layer-idx', type=int, default=2, dest='layer_idx',
                        help='layer index of the DraftPickNN to train on.')
    parser.add_argument('--draft-net-file', type=str, default=None, dest='draft_net_file',
                        help='File name of the DraftPickNN to train on.')
    args = parser.parse_args()
    if args.set_config_path is not None:
        _SET_CONFIG = set_config.get_set_config_from_path(args.set_config_path)
    else:
        _SET_CONFIG = set_config.get_set_config(args.set_id)
    draft_net = torch.load(args.draft_net_file, weights_only=False)
    train_and_save_sae_on_nnet_activation(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        input_dim=_SET_CONFIG.set_size,
        draft_net=draft_net,
        layer_idx=args.layer_idx)
