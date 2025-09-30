# **Installation**

## 1. Clone Repository

```bash
git clone <https://github.com/Tsutayaaa/Plaszyme.git>
cd Plaszyme
```
## 2. Create Conda Environment
We recommend using **[Conda](https://docs.conda.io/en/latest/)** to manage environments.

The general dependencies are provided in [`environment.yml`](./environment.yml). 

```bash
conda env create -f environment.yml
conda activate plaszyme_gnn
```

## 3.Install PyTorch
The deep learning backbone of this project is implemented using [PyTorch](https://pytorch.org/).

Install the correct version of PyTorch for your system.
Refer to the [PyTorch official installation guide](https://pytorch.org/get-started/locally/).

Example (CUDA 12.8):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
CPU-only (or Mac):
```bash
pip3 install torch torchvision
```

## 4.Install PyTorch Geometric (PyG)
[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/) is the core library we use to build graph-based protein and polymer representations.

Install following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## 5. Verify Installation
Check that PyTorch and PyG are installed correctly:
```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```
Expected output:
```
2.7.1 2.6.1
```
