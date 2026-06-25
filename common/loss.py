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


def weighted_quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus_hat: torch.Tensor,
    tau_deltas: torch.Tensor,
    kappa=1.0,
):
    pairwise_diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    abs_diff = pairwise_diff.abs()
    huber_loss = torch.where(
        abs_diff <= kappa,
        0.5 * pairwise_diff.square(),
        kappa * (abs_diff - 0.5 * kappa),
    )
    taus_hat_expanded = taus_hat.unsqueeze(2)
    quantile_weight = torch.abs(taus_hat_expanded - (pairwise_diff < 0).float())
    loss = (quantile_weight * huber_loss).mean(dim=2)
    loss = (loss * tau_deltas).sum(dim=1).mean()
    return loss
