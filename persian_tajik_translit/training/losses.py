"""Cycle-consistency loss and Ezafe auxiliary head for ByT5 transliteration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EzafeHead(nn.Module):
    """Binary classification head on encoder hidden states for Ezafe detection.

    Predicts whether each encoder position is followed by the Persian genitive
    clitic /e/ (Ezafe). Trained with BCE loss against DadmaTools silver labels.
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        ezafe_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.mlp(encoder_hidden).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            ezafe_labels.float(),
            reduction="none",
        )
        loss = (loss * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        return loss


class CycleLoss(nn.Module):
    """Cycle-consistency round-trip loss via straight-through Gumbel-Softmax.

    Forward direction: source → greedy output ŷ
    Backward direction: ŷ → reconstruct source

    The discrete argmax in ŷ is made differentiable via ST Gumbel-Softmax so
    gradients flow through the backward pass into the shared model.
    """

    def __init__(self, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        forward_logits: torch.Tensor,
        reverse_prefix_ids: torch.Tensor,
        reverse_prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cycle loss given pre-computed forward logits.

        Args:
            model: The T5 model (shared weights for both directions).
            input_ids: Original source token IDs [B, S].
            attention_mask: Source attention mask [B, S].
            forward_logits: Logits from forward pass [B, T, V].
            reverse_prefix_ids: Prefix tokens for reverse direction [B, P].
            reverse_prefix_mask: Attention mask for reverse prefix [B, P].

        Returns:
            Scalar cycle loss.
        """
        soft_one_hot = F.gumbel_softmax(forward_logits, tau=self.tau, hard=True)
        embedding_weight = model.shared.weight
        soft_embeds = torch.matmul(soft_one_hot, embedding_weight)

        cycle_output = model(
            inputs_embeds=soft_embeds,
            attention_mask=torch.ones(
                soft_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
            ),
            labels=input_ids,
        )
        return cycle_output.loss
