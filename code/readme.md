# SketchDreamer

### [Project Page](https://dreamgaussian.github.io) 
[![webpage](https://img.shields.io/badge/🌐-Website%20-blue.svg)](https://jdd-3d.github.io/SketchDreamer/) 

---

## Installation

> [!NOTE]
> Make sure Python, PyTorch, and the appropriate CUDA toolkit are available before installing the project dependencies.

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Gaussian Splatting rasterization

Install the modified Gaussian Splatting rasterizer with depth and alpha rendering support:

```bash
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

### 3. Install Simple KNN

```bash
pip install ./simple-knn
```

### 4. Install NVIDIA DiffRast

```bash
pip install git+https://github.com/NVlabs/nvdiffrast/
```

### 5. Install Kiuikit

```bash
pip install git+https://github.com/ashawkey/kiuikit
```

## Tested Environments

| Operating system | PyTorch | CUDA | GPU |
| :--- | :---: | :---: | :--- |
| Ubuntu 22 | 1.12 | 11.6 | NVIDIA V100 |
| Windows 10 | 2.1 | 12.1 | NVIDIA RTX 4090 |

---
