# dyGRASS

Code for dyGRASS — Dynamic Graph Spectral Sparsification via Localized Random Walks is the open-source reference implementation of the framework introduced in our ICCAD 2025 paper. dyGRASS updates spectral sparsifiers in O(1) time per edge insertion or deletion by launching GPU-accelerated non-backtracking random walks that pinpoint spectrally critical edges; this yields ~10 × speed-ups over the previous inGRASS algorithm while preserving tight spectral similarity across fully dynamic graphs.

## Setup

### System
* **Linux 22.04** (tested)  
* **CUDA 11.x or newer** with `nvcc` in `$PATH`

### 1. Grab the dataset
```bash
pip install  gdown         
python3 datasetdownload.py    # downloads + unpacks into ./dataset/     
```

### 2.Build the CUDA random-walk updaters

Prerequisite — CUDA toolkit 11.0 or newer (make sure `nvcc` is on your `$PATH`).

```bash
# Compile the incremental updater
cd random_walk_incremental
nvcc main.cu functions.cpp -o incremental_update
cd ..

# Compile the decremental updater
cd ../random_walk_decremental
nvcc main.cu functions.cpp -o decremental_update
cd ..
```
### 3. Julia environment for result reproduction
Open Julia in the project root and add the required packages:
```bash
using Pkg
Pkg.add([
    "Laplacians",
    "SparseArrays",
    "LinearAlgebra",
    "Arpack",
    "JLD2",
    "IterativeSolvers",
    "Statistics",
])
```

## USage

```bash
./random_walk_incremental/incremental_update <graph_name> <distortion_threshold> <#walkers>
# example
./random_walk_incremental/incremental_update G2 10 512
```
- graph_name  base name of the graph files inside dataset/
- distortion_threshold spectral-similarity bound (e.g. 100)
- walkers  number of non-backtracking walkers (≤ 512, multiple of 32)


```bash
./random_walk_decremental/decremental_update <graph_name>
# example
./random_walk_decremental/decremental_update 333SP
```

## Citing


```bibtex
@inproceedings{yuan2025dygrass,
  author    = {Yihang Yuan, Ali Aghdaei and Zhuo Feng},
  title     = {{dyGRASS}: Dynamic Spectral Graph Sparsification via Localized Random Walks on GPUs},
  booktitle = {Proceedings of the IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  year      = {2025}
}
```

## License

dyGRASS is released under the **MIT License**.  
See \`LICENSE\` for the full text.