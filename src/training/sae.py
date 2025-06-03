import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, l1_coeff=1e-5, sparsity_target=0.05, kl_coeff=1e-3):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l1_coeff = l1_coeff
        self.sparsity_target = sparsity_target # rho
        self.kl_coeff = kl_coeff # beta for KL divergence

        # Encoder
        self.encoder_fc = nn.Linear(input_dim, latent_dim)
        self.encoder_act = nn.ReLU() 

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        # Encode
        latent = self.encoder_act(self.encoder_fc(x))
        # Decode
        reconstructed = self.decoder_fc(latent)
        return reconstructed, latent

    def kl_divergence_loss(self, latent_activations):
        if self.kl_coeff == 0:
            return torch.tensor(0.0, device=latent_activations.device)

        # Average activation of each latent neuron across the batch
        rho_hat = torch.mean(latent_activations, dim=0) # Shape: [latent_dim]

        # Add epsilon for numerical stability (to avoid log(0))
        eps = 1e-7
        rho_hat = torch.clamp(rho_hat, eps, 1 - eps)

        kl_div = self.sparsity_target * torch.log(self.sparsity_target / rho_hat) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - rho_hat))

        return self.kl_coeff * torch.sum(kl_div) # Sum over latent dimensions

    def l1_sparsity_loss(self, latent_activations):
        if self.l1_coeff == 0:
            return torch.tensor(0.0, device=latent_activations.device)
        return self.l1_coeff * torch.mean(torch.abs(latent_activations)) # Mean over all elements in batch
