<div align="center">

<h1>SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection</h1>

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS%202026-blue?style=flat-square&logo=ieee)](https://doi.org/10.1109/TGRS.2026.3676658)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTGRS.2026.3676658-orange?style=flat-square)](https://doi.org/10.1109/TGRS.2026.3676658)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Mamba](https://img.shields.io/badge/mamba--ssm-2.2.4-8B5CF6?style=flat-square)](https://github.com/state-spaces/mamba)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**Tanishq Rachamalla\*, Aryan Das\*, Swalpa Kumar Roy, Swagatam Das, Lorenzo Bruzzone**

*\* Equal contribution*

*IEEE Transactions on Geoscience and Remote Sensing, 2026*

[![Read Paper](https://img.shields.io/badge/Read%20Paper-PDF-D32F2F?style=flat-square&logo=adobeacrobatreader&logoColor=white)](assets/SM-HAD_paper.pdf)

</div>

---

## Overview

**SM-HAD** is a unified encoder–decoder framework for **unsupervised hyperspectral anomaly detection (HAD)**. It integrates three complementary modules — Fourier-based spectral decomposition, locality-aware masked attention, and linear-time state-space modeling — into a single hybrid architecture that jointly captures:

- **Spectral diversity** via orthogonal frequency-domain decomposition
- **Spatial locality** via diagonal self-suppression masking
- **Long-range dependencies** via Mamba state-space modeling with O(L) complexity

Evaluated on **six benchmark datasets** against **18 state-of-the-art methods**, SM-HAD achieves the **highest average AUC of 0.9921** while requiring only **0.28M parameters** and **1.46 GFLOPs** — the best accuracy-efficiency trade-off among all compared methods.

---

## Architecture

<div align="center">
  <img src="assets/Architecture.png" width="100%">
  <p><em>SM-HAD pipeline: self-supervised training stage (top) and anomaly detection stage (bottom). Sub-modules: (b) OSFB, (c) Residual Mamba, (d) MLP, (e) MVAB.</em></p>
</div>

The architecture consists of three core modules inside a **Hybrid Transformer Block**:

### 1. Ortho Spectrum Fourier Block (OSFB)
Projects patch features into the 2-D frequency domain via orthonormalised FFT. Two-stage learnable complex filtering (nonlinear ReLU stage + linear refinement stage) selectively amplifies discriminative spectral modes. Soft-shrinkage sparsification suppresses noisy coefficients, and IFFT reconstructs the enhanced spatial representation. This captures global spectral correlations that span all spectral bands simultaneously.

### 2. Masked Vanilla Attention Block (MVAB)
Applies a fixed diagonal self-suppression mask (M<sub>a,a</sub> = −∞) that prevents each patch from attending to itself. A second-order feedback mechanism refines attention weights using outer-product self-similarity. This preserves fine-grained spatial locality and avoids the over-smoothing that typically blurs subtle anomaly boundaries in dense attention.

### 3. Residual Mamba Block (RMB)
Uses a structured state-space model (Mamba) with **linear complexity O(L·D)** to capture long-range spatial dependencies across the entire patch sequence. A depthwise 1-D convolution enriches local context before the global SSM scan, and a residual connection ensures stable gradient flow through deep spectral feature spaces.

---

## Key Contributions

1. **First unified hybrid HAD framework** integrating Fourier spectral modeling, locality-aware masked attention, and linear state-space learning in a single encoder–decoder.
2. **OSFB**: Orthogonal frequency-domain decomposition with two-stage complex filtering (nonlinear + linear) and soft-shrinkage for global spectral representation without quadratic cost.
3. **MVAB**: Binary diagonal masking that preserves spatial neighborhood structure, reduces over-smoothing, and improves fine-grained anomaly boundary localization.
4. **RMB**: Linear-time O(L) Mamba SSM for global context modeling with stable training and low computational overhead.
5. **0.9921 average AUC** on six benchmarks — best across all 19 compared methods while using fewer parameters and lower FLOPs than every deep learning baseline.

---

## Repository Structure

```
SM-HAD/
├── Datasets/                         # Six benchmark HSI datasets (.mat)
│   ├── los-angeles-1.mat
│   ├── los-angeles-2.mat
│   ├── gulfport.mat
│   ├── texas-coast.mat
│   ├── cat-island.mat
│   └── pavia.mat
├── assets/
│   └── pictures/
│       └── Architecture.png  # Architecture figure
├── results/                          # Generated: anomaly maps + ROC curves
├── logs/                             # Generated: per-run training logs
├── net.py                            # Network (OSFB, MVAB, RMB, Net)
├── block.py                          # Patch extraction, folding, background search
├── data.py                           # HSI patch dataset loader
├── main.py                           # Training & evaluation entry point
├── utils.py                          # Seeding, parameter, normalisation helpers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Environment

Developed and tested with:

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.0+ |
| mamba-ssm | **2.2.4** |
| CUDA | 11.8 |
| GPU | NVIDIA RTX 3090 |

Install all dependencies:

```bash
pip install -r requirements.txt
```

For `mamba-ssm`, a CUDA-capable GPU is required. If you face installation issues:

```bash
pip install mamba-ssm==2.2.4 
```

---

## Datasets

All six benchmark datasets are sourced from the [GT-HAD benchmark](https://github.com/swalpa/GT-HAD) and are included in the `Datasets/` directory.

| Dataset | Sensor | Spatial | Bands | Resolution |
|---------|--------|---------|-------|------------|
| Los Angeles-1 | AVIRIS | 100×100 | 205 | 7.1 m |
| Los Angeles-2 | AVIRIS | 100×100 | 205 | 7.1 m |
| Gulfport | AVIRIS | 100×100 | 191 | 3.4 m |
| Texas Coast | AVIRIS | 100×100 | 204 | 17.2 m |
| Cat Island | AVIRIS | 150×150 | 188 | 17.2 m |
| Pavia | ROSIS | 150×150 | 102 | 1.3 m |

Each `.mat` file must contain:
- `data` — HSI cube of shape `(H, W, C)`
- `groundtruth` — binary anomaly mask of shape `(H, W)`

---

## Training & Evaluation

Run SM-HAD on all six datasets with 3 independent seeds:

```bash
python main.py
```

This will:
1. Load and normalise each dataset from `Datasets/`
2. Train SM-HAD for 100 epochs (Adam, lr=1e-3, batch=64)
3. Run 3 independent experiments per dataset
4. Save best anomaly maps and ROC data to `results/SM-HAD/<dataset>/`
5. Write an Excel summary to `results/SM-HAD/SM-HAD-results.xlsx`
6. Write training logs to `logs/`


---

## Results

### AUC Comparison on Six Benchmark Datasets

| Method Type | Method | Paper | Code | LA-1 | LA-2 | Gulfport | Texas Coast | Cat Island | Pavia | **Avg AUC** |
|---|---|:---:|:---:|---|---|---|---|---|---|---|
| Statistical | RX | [IEEE'90](https://ieeexplore.ieee.org/document/6788565) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.8221 | 0.8404 | 0.9526 | 0.9907 | 0.9807 | 0.9538 | 0.9234 |
| | 2S-GLRT | [SigPro'20](https://doi.org/10.1016/j.sigpro.2020.107785) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.9236 | 0.9873 | 0.9183 | 0.9970 | 0.9995 | 0.9867 | 0.9687 |
| | MsRFQFT | [TGRS'21](https://ieeexplore.ieee.org/document/9321551) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.9114 | 0.9798 | 0.9945 | 0.9953 | 0.9798 | 0.9984 | 0.9765 |
| Representation | KIFD | [TGRS'20](https://ieeexplore.ieee.org/document/8976509) | [PuhongDuan](https://github.com/PuhongDuan/Hyperspectral-anomaly-detection-with-kernel-isolation-forest) | 0.9359 | 0.9775 | 0.9728 | 0.9303 | 0.9900 | 0.8634 | 0.9450 |
| | CRD | [TGRS'15](https://ieeexplore.ieee.org/document/7024907) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.9212 | 0.9681 | 0.9445 | 0.9940 | 0.9946 | 0.9407 | 0.9605 |
| | GTVLRR | [TGRS'21](https://ieeexplore.ieee.org/document/9416235) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.9004 | 0.8840 | 0.9874 | 0.9478 | 0.9694 | 0.9747 | 0.9440 |
| | PTA | [TGRS'19](https://ieeexplore.ieee.org/document/8693751) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.8809 | 0.9104 | 0.9946 | 0.9757 | 0.9831 | 0.9749 | 0.9533 |
| | PCA-TLRSR | [TCYB'22](https://ieeexplore.ieee.org/document/9781337) | [GT-HAD repo](https://github.com/jeline0110/GT-HAD) | 0.9455 | 0.9664 | 0.9930 | 0.9923 | 0.9854 | 0.9635 | 0.9744 |
| Deep Learning | LREN | [AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/16536) | [xdjiangkai](https://github.com/xdjiangkai/LREN) | 0.7327 | 0.9134 | 0.8293 | 0.9783 | 0.9343 | 0.8985 | 0.8811 |
| | BS3LNet | [TGRS'23](https://ieeexplore.ieee.org/document/10070799) | [DegangWang97](https://github.com/DegangWang97/IEEE_TGRS_BS3LNet) | 0.8550 | 0.8424 | 0.9467 | 0.9583 | 0.9847 | 0.9593 | 0.9244 |
| | DirectNet | [TGRS'24](https://ieeexplore.ieee.org/document/10400466) | [DegangWang97](https://github.com/DegangWang97/IEEE_TGRS_DirectNet) | 0.8917 | 0.8912 | 0.9598 | 0.9896 | 0.9841 | 0.9878 | 0.9507 |
| | PUNNet | [GRSL'24](https://ieeexplore.ieee.org/document/10648847) | [DegangWang97](https://github.com/DegangWang97/IEEE_GRSL_PUNNet) | 0.8755 | 0.9056 | 0.9652 | 0.9819 | 0.9792 | 0.9914 | 0.9498 |
| | Auto-HAD | [TGRS'21](https://ieeexplore.ieee.org/document/9382262) | [RSIDEA-WHU2020](https://github.com/RSIDEA-WHU2020/Auto-AD) | 0.9207 | 0.9063 | 0.9672 | 0.9909 | 0.9783 | 0.9807 | 0.9574 |
| | PDBSNet | [TGRS'23](https://ieeexplore.ieee.org/document/10129044) | [DegangWang97](https://github.com/DegangWang97/IEEE_TGRS_PDBSNet) | 0.9005 | 0.9100 | 0.9905 | 0.9913 | 0.9926 | 0.9887 | 0.9623 |
| | NL2Net | [JSTARS'25](https://ieeexplore.ieee.org/document/10858640) | [DegangWang97](https://github.com/DegangWang97/IEEE_JSTARS_NL2Net) | 0.9235 | 0.9209 | 0.9876 | 0.9702 | 0.9914 | 0.9843 | 0.9630 |
| | GT-HAD | [TNNLS'24](https://ieeexplore.ieee.org/document/10432978) | [jeline0110](https://github.com/jeline0110/GT-HAD) | 0.9515 | 0.9666 | 0.9888 | 0.9969 | 0.9984 | 0.9993 | 0.9836 |
| | HTD-Mamba | [TGRS'25](https://ieeexplore.ieee.org/document/10908894) | [shendb2022](https://github.com/shendb2022/HTD-Mamba) | 0.9465 | **0.9942** | **0.9978** | 0.9961 | 0.9907 | 0.9800 | 0.9842 |
| | SGLNet | [IPM'26](https://doi.org/10.1016/j.ipm.2025.104154) | [xautzhaozhe](https://github.com/xautzhaozhe/SGLNet) | 0.9377 | 0.9844 | 0.9915 | 0.9905 | 0.9952 | 0.9950 | 0.9824 |
| **Proposed** | **SM-HAD** | [TGRS'26](https://doi.org/10.1109/TGRS.2026.3676658) | **This repo** | **0.9793** | 0.9833 | 0.9929 | **0.9982** | **0.9996** | **0.9994** | **0.9921** |

*Bold = best. All experiments: NVIDIA RTX 3090, Python 3.10, PyTorch 2.0.*

### Model Complexity Comparison

| Model | Paper | Code | FLOPs | Params | Runtime (LA-1) |
|-------|:---:|:---:|-------|--------|----------------|
| [LREN](https://ojs.aaai.org/index.php/AAAI/article/view/16536) | AAAI'21 | [GitHub](https://github.com/xdjiangkai/LREN) | 11.97 G | 3.25 M | 30.55 s |
| [Auto-HAD](https://ieeexplore.ieee.org/document/9382262) | TGRS'21 | [GitHub](https://github.com/RSIDEA-WHU2020/Auto-AD) | 0.19 G | 0.86 M | 161.85 s |
| [GT-HAD](https://ieeexplore.ieee.org/document/10432978) | TNNLS'24 | [GitHub](https://github.com/jeline0110/GT-HAD) | 2.64 G | 0.26 M | 29.54 s |
| [HTD-Mamba](https://ieeexplore.ieee.org/document/10908894) | TGRS'25 | [GitHub](https://github.com/shendb2022/HTD-Mamba) | 0.13 G | 0.32 M | 549.02 s |
| [SGLNet](https://doi.org/10.1016/j.ipm.2025.104154) | IPM'26 | [GitHub](https://github.com/xautzhaozhe/SGLNet) | 5.46 G | 0.55 M | 54.00 s |
| **SM-HAD (Ours)** | **TGRS'26** | **This repo** | **1.46 G** | **0.28 M** | **25.65 s** |

---

## Reproducibility

- **Seeds**: 3 independent runs per dataset using seeds 0, 100, 200
- **Reported metric**: best AUC across 3 runs (consistent with paper)
- **No ground-truth labels** used at any training stage (fully unsupervised)
- **Hardware**: NVIDIA RTX 3090, CUDA 11.8
- **Training time**: ~25 seconds per dataset
- Pretrained checkpoints are not distributed; training from scratch is fast

---

## Citation

If you find SM-HAD useful in your research, please cite:

```bibtex
@ARTICLE{RachamallaSMHAD,
  author  = {Rachamalla, Tanishq and Das, Aryan and Roy, Swalpa Kumar
             and Das, Swagatam and Bruzzone, Lorenzo},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {1-1},
  doi     = {10.1109/TGRS.2026.3676658}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
This code is provided for academic and research use. Commercial use requires prior written permission from the authors.

