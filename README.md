# imgtools
Tools for Single-Cell 3D Imaging Data Analysis

## Installation

We recommend using Python 3.11:
```bash
conda create -n img python=3.11 -y
conda activate img
```

First, make sure to install the alabtools library (https://github.com/alberlab/alabtools).

Then, install with conda-forge the following dependencies:
```bash
conda install -c conda-forge \
    trimesh \
    scikit-learn \
    pydantic \
    imbalanced-learn \
    xgboost \
    libspatialindex \
    rtree \
    statsmodels \
    mrcfile \
    alphashape \
    -y
```

Finally, install imgtools:
```bash
pip install git+https://github.com/fmusella/imgtools.git
```
