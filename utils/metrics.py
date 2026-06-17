import torch

def si_snr(estimate, target):
    """
    Tính Scale-Invariant Signal-to-Noise Ratio.
    Args:
        estimate: (batch, T) hoặc (T,)
        target:   (batch, T) hoặc (T,)
    Returns:
        SI-SNR: scalar
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target   = target   - target.mean(dim=-1, keepdim=True)

    dot      = (estimate * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
    s_target = dot / s_target_energy * target

    noise    = estimate - s_target
    ratio    = (s_target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
    return 10 * torch.log10(ratio + 1e-8)


def pit_si_snr(estimates, targets):
    """
    Permutation Invariant Training với SI-SNR.
    Args:
        estimates: (batch, num_spks, T)
        targets:   (batch, num_spks, T)
    Returns:
        loss: scalar (âm của SI-SNRi trung bình)
    """
    batch, num_spks, T = estimates.shape
    best_loss = None

    from itertools import permutations
    for perm in permutations(range(num_spks)):
        loss = 0
        for i, j in enumerate(perm):
            loss += si_snr(estimates[:, i, :], targets[:, j, :])
        loss = loss / num_spks
        if best_loss is None:
            best_loss = loss
        else:
            best_loss = torch.max(best_loss, loss)

    return -best_loss.mean()