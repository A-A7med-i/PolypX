from src.config.constant import FILM_LAYER
import segmentation_models_pytorch as smp
from typing import List, Tuple
import torch.nn as nn
import torch


class FiLMGenerator(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) generator module.

    This module generates scaling (gamma) and shifting (beta) parameters
    based on a sequence ID, which are then used to modulate feature maps
    in a neural network.

    Args:
        num_sequences (int): The total number of unique sequences.
        emb_dim (int): Dimensionality of the sequence embedding.
        num_channels (int): Number of channels in the feature map to be modulated.
    """

    def __init__(self, num_sequences: int, emb_dim: int, num_channels: int):
        super().__init__()
        self.embedding = nn.Embedding(num_sequences + 1, emb_dim)

        self.gamma_fc = nn.Linear(emb_dim, num_channels)
        self.beta_fc = nn.Linear(emb_dim, num_channels)

    def forward(self, seq_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM scaling (gamma) and shifting (beta) parameters.

        Args:
            seq_id (torch.Tensor): Sequence IDs of shape ``[B]``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - **gamma**: Scaling parameters of shape ``[B, C]``.
                - **beta**: Shifting parameters of shape ``[B, C]``.
        """
        seq_vec = self.embedding(seq_id)

        raw_gamma = self.gamma_fc(seq_vec)
        raw_beta = self.beta_fc(seq_vec)

        gamma = 1.0 + 0.1 * torch.tanh(raw_gamma)
        beta = 0.1 * torch.tanh(raw_beta)
        return gamma, beta

    def apply_film(
        self, features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to feature maps.

        Args:
            features (torch.Tensor): Feature maps of shape ``[B, C, H, W]``.
            gamma (torch.Tensor): Scaling parameters of shape ``[B, C]``.
            beta (torch.Tensor): Shifting parameters of shape ``[B, C]``.

        Returns:
            torch.Tensor: FiLM-modulated feature maps of shape ``[B, C, H, W]``.
        """
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C] -> [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B, C] -> [B, C, 1, 1]
        return gamma * features + beta


class PolypFiLMNet(nn.Module):
    """
    A deep learning model for polyp segmentation that incorporates FiLM.

    The model uses a pre-trained DeepLabV3Plus backbone and modulates
    an intermediate encoder layer with FiLM parameters conditioned on
    a sequence identifier. This allows the network to adapt predictions
    to sequence-specific context.

    Args:
        num_sequences (int): Total number of unique sequences.
        emb_dim (int): Dimensionality of the sequence embedding for FiLM.
    """

    def __init__(self, num_sequences, emb_dim):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=1,
        )

        num_channels = self.backbone.encoder.out_channels[FILM_LAYER]

        self.film_generator = FiLMGenerator(
            num_sequences=num_sequences, emb_dim=emb_dim, num_channels=num_channels
        )

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor of shape ``[B, 3, H, W]``.
            seq_id (torch.Tensor): Sequence ID tensor of shape ``[B]``.

        Returns:
            torch.Tensor: Output segmentation mask tensor of shape ``[B, 1, H, W]``.
        """

        features: List[torch.Tensor] = self.backbone.encoder(x)

        gamma, beta = self.film_generator(seq_id)

        features[FILM_LAYER] = self.film_generator.apply_film(
            features[FILM_LAYER], gamma, beta
        )

        out = self.backbone.decoder(features)
        out = self.backbone.segmentation_head(out)
        return out
