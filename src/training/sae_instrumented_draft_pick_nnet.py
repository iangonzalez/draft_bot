import torch.nn as nn
import copy

class SAEInstrumentedDraftPickNN(nn.Module):
    """
    NNet for picking cards in a draft.
    Modified to optionally insert a Sparse Autoencoder (SAE)
    to replace the output of a dropout layer.
    """

    def __init__(self, draft_net, sae_model=None, sae_replaces_dropout_output_idx=None):
        """
        INPUTS:
        - draft_net: The DraftPickNN to copy weights from. This will be copied.
        - sae_model (Optional[nn.Module]): An instantiated, pre-trained SparseAutoencoder.
        - sae_replaces_dropout_output_idx (Optional[int]): Index of the dropout layer whose
            output will be replaced by the SAE's encoded features.
            - 0: Replaces output of self.dropout1 (input to self.linear2)
            - 1: Replaces output of self.dropout2 (input to self.linear3)
            - 2: Replaces output of self.dropout3 (input to self.linear4)
        """
        super().__init__()
        self.draft_net_copy = copy.deepcopy(draft_net)
        self.draft_net_copy.eval()
        self.sae_model = sae_model
        self.sae_replaces_dropout_output_idx = sae_replaces_dropout_output_idx
        self.latent_activations = {"layer1": None, "layer2": None, "layer3": None}
        self.pre_latent_hidden_activations = {"layer1": None, "layer2": None, "layer3": None}
        self.latent_multipliers = {}

        if self.sae_model is not None:
            if self.sae_model.input_dim != self.draft_net_copy.net_dim:
              raise AttributeError("SAE input dim mismatch")

            # Freeze the SAE model
            self.sae_model.eval()
            for param in self.sae_model.parameters():
                param.requires_grad = False


    def get_most_recent_hidden_activations(self):
        return self.pre_latent_hidden_activations

    def get_most_recent_latent_activations(self):
        return self.latent_activations
    
    def get_most_recent_prepack_activations(self):
        return self.pre_pack_activations

    def set_latent_multipliers(self, latents, multipliers):
        assert len(latents) == len(multipliers)
        for l, m in zip(latents, multipliers):
            self.latent_multipliers[l] = m

    def clear_latent_multipliers(self):
        self.latent_multipliers = {}

    def run_sae_with_latent_multipliers(self, x):
        # Encode
        latent = self.sae_model.encoder_act(self.sae_model.encoder_fc(x))
        for idx, multiplier in self.latent_multipliers.items():
            latent[:, idx] *= multiplier
        # Decode
        reconstructed = self.sae_model.decoder_fc(latent)
        return reconstructed, latent
    
    def _maybe_apply_sae_to_layer(self, x, layer_idx):
        if self.sae_replaces_dropout_output_idx == layer_idx and self.sae_model:
            if self.latent_multipliers:
                reconstructed, latent = self.run_sae_with_latent_multipliers(x)
            else:
                reconstructed, latent = self.sae_model(x)
            self.latent_activations[f"layer{layer_idx}"] = latent
            return reconstructed
        else:
            return x

    def forward(self, x):
        pack = x[:, :self.draft_net_copy.net_dim]
        pool = x[:, self.draft_net_copy.net_dim:]

        # --- Block 1 ---
        y = self.draft_net_copy.linear1(pool)
        y = self.draft_net_copy.bn1(y)
        y = self.draft_net_copy.relu1(y)
        y_after_dropout1 = self.draft_net_copy.dropout1(y)
        self.pre_latent_hidden_activations["layer1"] = y_after_dropout1.clone().detach()

        y_for_linear2 = self._maybe_apply_sae_to_layer(y_after_dropout1, 1)

        # --- Block 2 ---
        y = self.draft_net_copy.linear2(y_for_linear2)
        y = self.draft_net_copy.bn2(y)
        y = self.draft_net_copy.relu2(y)
        y_after_dropout2 = self.draft_net_copy.dropout2(y)
        self.pre_latent_hidden_activations["layer2"] = y_after_dropout2.clone().detach()

        y_for_linear3 = self._maybe_apply_sae_to_layer(y_after_dropout2, 2)

        # --- Block 3 ---
        y = self.draft_net_copy.linear3(y_for_linear3)
        y = self.draft_net_copy.bn3(y)
        y = self.draft_net_copy.relu3(y)
        y_after_dropout3 = self.draft_net_copy.dropout3(y)
        self.pre_latent_hidden_activations["layer3"] = y_after_dropout3.clone().detach()

        y_for_linear4 = self._maybe_apply_sae_to_layer(y_after_dropout3, 3)

        # --- Final Linear Layer ---
        y = self.draft_net_copy.linear4(y_for_linear4)
        self.pre_pack_activations = y.clone().detach()

        y *= pack # Apply pack constraint
        return y
