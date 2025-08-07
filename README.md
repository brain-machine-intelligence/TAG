# TAG — Topology‑Aware Grid Coding

Code and data for the manuscript
**“Topological spatial coding for rapid generalization in the hippocampal formation”**

---

## Quick‑start

### 1. Install Git LFS before cloning

Git Large File Storage is required to fetch large data files.

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install git-lfs

# Or via conda
conda install -c conda-forge git-lfs

# Enable LFS for your user account
git lfs install
```

### 2. Clone the repository

```bash
git clone https://github.com/brain-machine-intelligence/TAG.git
cd TAG
```

### 3. Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate tag
```

---

## Usage

Run the end‑to‑end pipeline:

```bash
python main.py
```

---

## Reproducing figures

Each figure has a companion notebook.
Open the notebook you’re interested in and **Run All** after the main pipeline completes.

---
