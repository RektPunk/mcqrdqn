import torch


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    quantiles: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    pairwise_diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    abs_diff = pairwise_diff.abs()
    huber_loss = torch.where(
        abs_diff <= kappa,
        0.5 * pairwise_diff.square(),
        kappa * (abs_diff - 0.5 * kappa),
    )

    quantile_weight = torch.abs(
        quantiles.unsqueeze(2) - (pairwise_diff.detach() < 0).float()
    )
    loss = quantile_weight * huber_loss
    return loss.mean(dim=2).sum(dim=1).mean()
