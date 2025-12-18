import torch


def window_partition(x: torch.Tensor, window_size, device):
    """
    Args:
        x: (B, D, N, C)
        window_size (int,int): Must be a power of 2 in the healpy grid.

    Returns:
        windows: (num_windows*B, window_size_d * window_size_hp , C)
    """
    B, D, N, C = x.shape
    window_size_d, window_size_hp = window_size
    x = x.view(
        B, D // window_size_d, window_size_d, N // window_size_hp, window_size_hp, C
    )
    # B, D//wd, wd, N//whp, whp, c
    # 0  1      2   3       4    5
    # =>
    # B, D//wd, N//whp, wd, whp, c
    # 0  1      3       2   4    5
    x = x.permute(0, 1, 3, 2, 4, 5)
    windows = x.contiguous().view(-1, window_size_d * window_size_hp, C)
    return windows


def window_reverse(windows, window_size, D, N, device):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Must be a power of 2 in the healpy grid
        N (int): Number of pixels in the healpy grid

    Returns:
        x: (B, N, C)
    """
    window_size_d, window_size_hp = window_size

    B = int(windows.shape[0] / (D * N // (window_size_hp * window_size_d)))
    x = windows.view(
        B, D // window_size_d, N // window_size_hp, window_size_d, window_size_hp, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.contiguous().view(B, D, N, -1)
    return x
