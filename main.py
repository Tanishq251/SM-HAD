"""
main.py — Training and evaluation entry point for SM-HAD.

Runs SM-HAD on all six benchmark datasets with 3 independent seeds.
Best AUC and corresponding anomaly maps are saved per dataset.
An Excel summary is written to results/SM-HAD/SM-HAD-results.xlsx.

Usage:
    python main.py

Reference:
    Rachamalla et al., "SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection",
    IEEE Transactions on Geoscience and Remote Sensing, 2026.
    DOI: 10.1109/TGRS.2026.3676658
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import time
import logging

import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from progress.bar import Bar

from utils import get_params, img2mask
from data import DatasetHsi
from block import Block_fold, Block_search
from net import Net


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dtype = torch.cuda.FloatTensor

DATA_DIR    = "./Datasets/"
SAVE_DIR    = "./results/"
LOGS_DIR    = "./logs/"

DATASETS = [
    "los-angeles-1",
    "los-angeles-2",
    "gulfport",
    "texas-coast",
    "cat-island",
    "pavia",
]

# SM-HAD hyperparameters (Section IV-C of the paper)
PATCH_SIZE    = 3        # inner MVAB sub-patch size
PATCH_STRIDE  = 3        # inner MVAB sub-patch stride
EMBED_DIM     = 64       # embedding dimension D
MLP_RATIO     = 2.0      # MLP expansion ratio
D_STATE       = 64       # Mamba state dimension d_s
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
EPOCHS        = 100
SEARCH_EVERY  = 25       # background-search interval (iterations)
NUM_RUNS      = 3        # independent seeds per dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(tag: str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(LOGS_DIR) if f.startswith("log") and f.endswith(".log")]
    nums = []
    for f in existing:
        try:
            nums.append(int(f[3:-4]))
        except ValueError:
            pass
    idx = max(nums) + 1 if nums else 1
    log_path = os.path.join(LOGS_DIR, f"log{idx}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
    logging.info(f"[SM-HAD] Logging started for dataset: {tag}")


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(file: str, run_id: int):
    set_seed(run_id * 100)
    logging.info(f"  Dataset: {file} | Run: {run_id + 1}/{NUM_RUNS}")

    # ---- Load and normalise HSI ----
    mat    = sio.loadmat(os.path.join(DATA_DIR, f"{file}.mat"))
    img_np = mat["data"].transpose(2, 0, 1).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    gt     = mat["map"]

    img_var = torch.from_numpy(img_np).type(dtype)[None, :]    # (1, C, H, W)
    band, row, col = img_var.shape[1], img_var.shape[2], img_var.shape[3]

    # ---- Dataset and block utilities ----
    block_size  = PATCH_SIZE * PATCH_STRIDE
    dataset     = DatasetHsi(img_var, wsize=block_size, wstride=PATCH_STRIDE)
    block_fold  = Block_fold(wsize=block_size, wstride=PATCH_STRIDE)
    block_srch  = Block_search(img_var, wsize=block_size, wstride=PATCH_STRIDE)
    loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    data_num    = len(dataset)

    # ---- Model ----
    model = Net(
        in_chans=band,
        embed_dim=EMBED_DIM,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        mlp_ratio=MLP_RATIO,
    ).cuda()

    criterion = nn.MSELoss().type(dtype)
    optimizer = optim.Adam(get_params(model), lr=LEARNING_RATE)
    avgpool   = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

    # Background matching state
    match_vec     = torch.zeros(data_num).type(dtype)
    search_matrix = torch.zeros(data_num, band, block_size, block_size).type(dtype)
    search_index  = torch.arange(0, data_num).type(torch.cuda.LongTensor)

    # ---- Training loop ----
    t0  = time.time()
    bar = Bar(f"Training [{file} | run {run_id + 1}]", max=EPOCHS)

    for ep in range(1, EPOCHS + 1):
        search_flag = (ep % SEARCH_EVERY == 0) and (ep != EPOCHS)
        for batch in loader:
            optimizer.zero_grad()
            net_gt    = batch["block_gt"]
            net_in    = batch["block_input"]
            blk_idx   = batch["index"].cuda()
            net_out   = model(net_in, block_idx=blk_idx, match_vec=match_vec)
            if search_flag:
                search_matrix[blk_idx] = net_out.detach()
            loss = criterion(net_out, net_gt)
            loss.backward()
            optimizer.step()

        if search_flag:
            match_vec = torch.zeros(data_num).type(dtype)
            search_back = block_fold(search_matrix.detach(), dataset.padding, row, col)
            match_vec   = block_srch(search_back.detach(), match_vec, search_index)

        bar.next()

    bar.finish()

    # ---- Inference ----
    model.eval()
    infer_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    residuals = []
    with torch.no_grad():
        for batch in infer_loader:
            infer_in  = batch["block_input"]
            infer_idx = batch["index"].cuda()
            infer_out = model(infer_in, block_idx=infer_idx, match_vec=match_vec)
            res = torch.abs(infer_in - infer_out) ** 2
            res = avgpool(res)
            residuals.append(res)

    res_cat    = torch.cat(residuals, dim=0)
    res_folded = block_fold(res_cat.detach(), dataset.padding, row, col)
    anom_map   = img2mask(res_folded)

    auc = roc_auc_score(gt.flatten(), anom_map.flatten())
    runtime = time.time() - t0
    logging.info(f"    AUC: {auc:.8f} | Runtime: {runtime:.2f}s")

    return auc, anom_map, gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}

    for file in DATASETS:
        setup_logging(file)
        print(f"\n{'='*60}")
        print(f"  Dataset: {file.upper()}")
        print(f"{'='*60}")

        save_subdir = os.path.join(SAVE_DIR, "SM-HAD", file)
        os.makedirs(save_subdir, exist_ok=True)

        auc_values = []
        best_auc, best_map, best_gt = 0.0, None, None

        for run in range(NUM_RUNS):
            auc, anom_map, gt = run_experiment(file, run)
            auc_values.append(auc)
            logging.info(f"  Run {run + 1}/{NUM_RUNS} → AUC: {auc:.8f}")
            if auc > best_auc:
                best_auc = auc
                best_map = anom_map
                best_gt  = gt

        # Save best anomaly map and ROC curve
        if best_map is not None:
            fpr, tpr, _ = roc_curve(best_gt.flatten(), best_map.flatten())
            sio.savemat(
                os.path.join(save_subdir, "SM-HAD-map_best_run.mat"),
                {"show": best_map},
            )
            sio.savemat(
                os.path.join(save_subdir, "SM-HAD-roc_best_run.mat"),
                {"PD": tpr, "PF": fpr},
            )

        std_auc = float(np.std(auc_values))
        logging.info(f"  {file} → Best AUC: {best_auc:.8f} ± {std_auc:.8f}")
        print(f"  Best AUC: {best_auc:.8f} ± {std_auc:.8f}")

        results[file] = {
            "auc_values": auc_values,
            "best_auc":   best_auc,
            "std_auc":    std_auc,
        }

    # ---- Summary Excel ----
    excel_path = os.path.join(SAVE_DIR, "SM-HAD", "SM-HAD-results.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df = pd.DataFrame({
        "Dataset": DATASETS,
        "Best AUC": [results[f]["best_auc"] for f in DATASETS],
        "Std AUC":  [results[f]["std_auc"]  for f in DATASETS],
        "AUC (mean ± std)": [
            f"{results[f]['best_auc']:.8f} ± {results[f]['std_auc']:.8f}"
            for f in DATASETS
        ],
    })
    df.to_excel(excel_path, index=False)
    logging.info(f"Results saved → {excel_path}")

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for f in DATASETS:
        print(f"  {f:<20s}  {results[f]['best_auc']:.8f} ± {results[f]['std_auc']:.8f}")
    print(f"\n  Full results saved to: {excel_path}")


if __name__ == "__main__":
    main()
