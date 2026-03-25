"""
block.py — Patch extraction, folding, and background-search utilities for SM-HAD.

Three modules:
    Block_embedding : unfold HSI into overlapping spatial patches
    Block_fold      : fold patches back to image space (averaging overlaps)
    Block_search    : match reconstructed patches to original to flag background
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block_embedding
# ---------------------------------------------------------------------------

class Block_embedding(nn.Module):
    """
    Extract overlapping spatial patches from a 4-D HSI tensor.

    Args:
        wsize   (int): patch (kernel) size in pixels
        wstride (int): extraction stride in pixels
    """

    def __init__(self, wsize: int = 15, wstride: int = 5):
        super().__init__()
        self.ksize = wsize
        self.stride = wstride

    def same_padding(self, images: torch.Tensor, ksizes, strides, rates=(1, 1)):
        """
        Compute and apply 'same' padding so the spatial dimensions tile evenly.

        Returns:
            images   : padded tensor
            paddings : tuple (left, right, top, bottom) used for fold-back
        """
        assert images.dim() == 4, "Expected (B, C, H, W)"
        _, _, rows, cols = images.shape
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        eff_k_row = (ksizes[0] - 1) * rates[0] + 1
        eff_k_col = (ksizes[1] - 1) * rates[1] + 1
        pad_rows = max(0, (out_rows - 1) * strides[0] + eff_k_row - rows)
        pad_cols = max(0, (out_cols - 1) * strides[1] + eff_k_col - cols)
        pad_top   = pad_rows // 2
        pad_left  = pad_cols // 2
        pad_bottom = pad_rows - pad_top
        pad_right  = pad_cols - pad_left
        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        images = F.pad(images, paddings, mode='replicate')
        return images, paddings

    def extract_image_blocks(self, images: torch.Tensor, ksizes, strides):
        images, paddings = self.same_padding(images, ksizes, strides)
        unfold = nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
        blocks = unfold(images)
        return blocks, paddings

    def forward(self, x: torch.Tensor):
        """
        Returns:
            blocks   : (N, C, ksize, ksize) patch tensor (gt and input are identical)
            paddings : tuple needed by Block_fold
        """
        _, band, _, _ = x.shape
        blocks, paddings = self.extract_image_blocks(
            x, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride]
        )
        blocks = blocks.squeeze(0).permute(1, 0)
        blocks = blocks.view(blocks.size(0), band, self.ksize, self.ksize)
        return blocks, blocks, paddings


# ---------------------------------------------------------------------------
# Block_fold
# ---------------------------------------------------------------------------

class Block_fold(nn.Module):
    """
    Fold a set of patches back into a full image by averaging overlapping regions.

    Args:
        wsize   (int): patch size (must match Block_embedding)
        wstride (int): stride (must match Block_embedding)
    """

    def __init__(self, wsize: int = 15, wstride: int = 5):
        super().__init__()
        self.ksize = wsize
        self.stride = wstride

    def forward(self, x: torch.Tensor, paddings, row: int, col: int):
        """
        Args:
            x        : (N, C, ksize, ksize) reconstructed patches
            paddings : padding tuple from Block_embedding.same_padding
            row, col : original image spatial dimensions

        Returns:
            Tensor : (1, C, H, W) folded image
        """
        num = x.size(0)
        back = x.view(num, -1).permute(1, 0).unsqueeze(0)

        block_size1 = (row + 2 * paddings[2] - (self.ksize - 1) - 1) / self.stride + 1
        block_size2 = (col + 2 * paddings[0] - (self.ksize - 1) - 1) / self.stride + 1
        pad = [paddings[3], paddings[1]] if block_size1 * block_size2 != num else [paddings[2], paddings[0]]

        ori = F.fold(back, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)

        # Normalise by overlap count
        ones = torch.ones_like(ori)
        ones_unfold = F.unfold(ones, (self.ksize, self.ksize), padding=pad, stride=self.stride)
        fold_mask = F.fold(ones_unfold, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)
        return ori / fold_mask


# ---------------------------------------------------------------------------
# Block_search
# ---------------------------------------------------------------------------

class Block_search(nn.Module):
    """
    For each reconstructed patch, find the nearest-neighbour original patch
    (L2 distance) and mark patches that match their own position as background.

    Args:
        x_ori   (Tensor): original HSI tensor (1, C, H, W)
        wsize   (int)   : patch size
        wstride (int)   : stride
    """

    def __init__(self, x_ori: torch.Tensor, wsize: int = 15, wstride: int = 5):
        super().__init__()
        self.ksize  = wsize
        self.stride = wstride
        self.dist   = nn.PairwiseDistance(p=2, keepdim=True)

        self.block_embedding = Block_embedding(wsize=wsize, wstride=wstride)
        block_query, _ = self.block_embedding.extract_image_blocks(
            x_ori, ksizes=[wsize, wsize], strides=[wstride, wstride]
        )
        self.block_query = block_query.squeeze(0).permute(1, 0)  # (N, C*k*k)

    def _pairwise_distance(self, x1: torch.Tensor, x2: torch.Tensor):
        """Compute row-wise L2 distance: x1 (B, D) vs x2 (N, D) → (B, N)."""
        return torch.cat(
            [self.dist(x1[i].unsqueeze(0), x2).unsqueeze(0) for i in range(x1.size(0))],
            dim=0
        ).squeeze(-1)

    def forward(self, x: torch.Tensor, match_vec: torch.Tensor, idx: torch.Tensor):
        """
        Args:
            x         : reconstructed folded image (1, C, H, W)
            match_vec : (N,) flag tensor — 1 = background patch
            idx       : (N,) patch indices being evaluated

        Returns:
            match_vec : updated flag tensor
        """
        block_key, _ = self.block_embedding.extract_image_blocks(
            x, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride]
        )
        block_key = block_key.squeeze(0).permute(1, 0)

        dis_map = self._pairwise_distance(block_key, self.block_query)
        _, nn_idx = torch.topk(dis_map, 1, dim=1, largest=False, sorted=True)
        nn_idx = nn_idx.squeeze()

        flag = (nn_idx == idx).int()
        match_vec[flag == 1] = 1
        return match_vec
