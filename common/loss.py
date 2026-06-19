import torch


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    quantiles: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Compute Quantile Huber Loss."""
    # pairwise_diff: (batch_size, r_target, r)
    pairwise_diff = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
    abs_diff = torch.abs(pairwise_diff)
    huber_loss = torch.where(
        abs_diff <= kappa,
        0.5 * pairwise_diff**2,
        kappa * (abs_diff - 0.5 * kappa),
    )
    loss = (
        torch.abs(quantiles - (pairwise_diff.detach() < 0).float()) * huber_loss / kappa
    )

    return loss.sum(dim=2).mean()
