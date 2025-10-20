# mmPoint: Dense Human Point Cloud Generation from mmWave

This repository is an implementation of the paper **"mmPoint: Dense Human Point Cloud Generation from mmWave"** by *Qian Xie et al.*, presented at the **British Machine Vision Conference (BMVC) 2023**.  
The model generates **dense 3D human point clouds** from sparse millimeter-wave (mmWave) radar signals, addressing sparsity and irregularity in mmWave data for applications like **surveillance, AR/VR, fitness tracking, and human pose estimation**.

Our implementation is based on the original codebase from [`NUAAXQ/mmPoint`](https://github.com/NUAAXQ/mmPoint).  
We have adapted it for improved setup, reproducibility, and potential extensions.

If you use this repository, please cite the original paper (details below).

---

## 🧠 Overview

mmWave radars provide 3D point clouds but typically yield **sparse outputs (64–128 points)** due to limited angular resolution.  
**mmPoint** reformulates point cloud generation as a **conditional deformation task**:

- **Input:** A single mmWave radar frame and a template human point cloud (e.g., 256 points).  
- **Output:** A dense point cloud (up to 2048 points) reflecting the human shape.

### Architecture Highlights

- **Multi-Modal Encoder (MME):** Extracts features from mmWave signals (using MNet) and point clouds (using EdgeConv).  
- **Multi-Resolution Decoder (MRD):** Employs a three-step **Lift-and-Deform Module (LDM)** to progressively densify and deform the point cloud.

### Key Contributions
- Novel task of **dense human point cloud generation from mmWave**.
- Multi-step deformation strategy for easier learning.
- New dataset with mmWave signals paired with pseudo-ground-truth point clouds from HuPR.

> ![Pipeline Overview](./docs/pipeline_overview.png)  
> *Figure: Overview of the mmPoint pipeline.*

---

## ✨ Features

- Generates **dense, uniform human point clouds** from single-frame mmWave signals.  
- Supports **training on the HuPR-derived dataset**.  
- Evaluation with **L1 Chamfer Distance**.  
- mmWave pre-processing (FFT for Doppler, Range, AoA).  
- Ablation for **one-step vs. three-step decoding**.  
- Visualization support (e.g., via **MeshLab**).

---

## ⚙️ Requirements

- Python 3.6+  
- PyTorch 1.10.2+  
- CUDA 11.3 (for GPU acceleration)  
- Other dependencies listed in `requirements.txt`  
  (includes NumPy, SciPy, h5py, tqdm, etc.)

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/RansadiDeAlwis/mmPoint.git
cd mmPoint

# Install dependencies
pip install -r requirements.txt
