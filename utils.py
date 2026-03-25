import torch

# Per-dataset random seed mapping
seed_dict = {
    'los-angeles-1': 0,
    'los-angeles-2': 0,
    'gulfport':      0,
    'cat-island':    0,
    'pavia':         0,
    'texas-coast':   0,
}


def get_params(net):
    """Return all learnable parameters of the network."""
    return list(net.parameters())


def img2mask(img):
    """
    Convert a folded residual tensor to a normalized 2D anomaly map.

    Args:
        img (Tensor): shape (1, C, H, W) — channel-summed reconstruction residual

    Returns:
        np.ndarray: normalized anomaly map in [0, 1], shape (H, W)
    """
    img = img[0].sum(0)
    img = img - img.min()
    img = img / img.max()
    return img.detach().cpu().numpy()
