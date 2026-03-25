"""
data.py — PyTorch Dataset for hyperspectral image patch loading.

DatasetHsi wraps Block_embedding to extract overlapping spatial patches
from an HSI cube at construction time, then serves them on-demand.
"""

import torch.utils.data as data
from block import Block_embedding


class DatasetHsi(data.Dataset):
    """
    Patch-based HSI dataset for self-supervised reconstruction training.

    At initialisation, the full HSI cube is divided into overlapping spatial
    patches using Block_embedding. Both the input and ground-truth are the
    same patch (self-supervised). Patch index is also returned so the
    attention mask can use it during training.

    Args:
        hsi_data (Tensor): shape (1, C, H, W), values normalised to [0, 1]
        wsize    (int)   : patch (kernel) size in pixels  — default 15
        wstride  (int)   : extraction stride in pixels    — default 5
    """

    def __init__(self, hsi_data, wsize: int = 15, wstride: int = 5):
        super().__init__()
        embedder = Block_embedding(wsize=wsize, wstride=wstride)
        self.block_gt, self.block_input, self.padding = embedder(hsi_data)

    def __getitem__(self, index: int):
        return {
            'block_gt':    self.block_gt[index],
            'block_input': self.block_input[index],
            'index':       index,
        }

    def __len__(self) -> int:
        return self.block_gt.size(0)
