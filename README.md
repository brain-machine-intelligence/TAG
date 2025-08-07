# TAG — Topology‑Aware Grid Coding

Code and data for the manuscript
**“Topological spatial coding for rapid generalization in the hippocampal formation”**

---

## Quick‑start

### 1. Install Git LFS

Large‑file support is necessary to retrieve large data files stored with LFS.

```bash
sudo apt update && sudo apt install git-lfs

git lfs install   # one‑time setup
```

### 2. Clone and set up the environment

```bash
git clone https://github.com/brain-machine-intelligence/TAG.git
cd TAG

conda env create -f environment.yaml
conda activate tag
```

---

## Data

The **`data/`** directory contains data required to reproduce the figures in the manuscript.
The script **`main.py`** documents the pipeline for generating the data.

---

## Figures

Each figure can be reproduced by executing its corresponding Jupyter notebook (e.g., `fig1.ipynb`, `fig2.ipynb`).

