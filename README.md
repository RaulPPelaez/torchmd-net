# TorchMD-NET

TorchMD-NET provides state-of-the-art neural networks potentials (NNPs) and a mechanism to train them. It offers efficient and fast implementations if several NNPs and it is integrated in GPU-accelerated molecular dynamics code like [ACEMD](https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/), [OpenMM](https://www.openmm.org) and [TorchMD](https://github.com/torchmd/torchmd). TorchMD-NET exposes its NNPs as [PyTorch](https://pytorch.org) modules.

## Available architectures

- [Equivariant Transformer (ET)](https://arxiv.org/abs/2202.02541)
- [Transformer (T)](https://arxiv.org/abs/2202.02541)
- [Graph Neural Network (GN)](https://arxiv.org/abs/2212.07492)
- [TensorNet](https://arxiv.org/abs/2306.06482)


## Installation

1. Clone the repository:
    ```
    git clone https://github.com/torchmd/torchmd-net.git
    cd torchmd-net
    ```

2. Install Mambaforge (https://github.com/conda-forge/miniforge/#mambaforge). We recommend to use `mamba` rather than `conda`.

3. Create an environment and activate it:
    ```
    mamba env create -f environment.yml
    mamba activate torchmd-net
    ```

4. Install TorchMD-NET into the environment:
    ```
    pip install -e .
    ```

## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. Several examples can be found in [examples/](https://github.com/torchmd/torchmd-net/tree/main/examples). Note that if a parameter is present both in the yaml file and the command line, the command line version takes precedence.
GPUs can be selected by setting the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1, the default, uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).
```
mkdir output
CUDA_VISIBLE_DEVICES=0 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml --log-dir output/
```

Run `torchmd-train --help` to see all available options and their descriptions.

## Pretrained models
See [here](https://github.com/torchmd/torchmd-net/tree/main/examples#loading-checkpoints) for instructions on how to load pretrained models.

## Creating a new dataset
If you want to train on custom data, first have a look at `torchmdnet.datasets.Custom`, which provides functionalities for 
loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels.
Alternatively, you can implement a custom class according to the torch-geometric way of implementing a dataset. That is, 
derive the `Dataset` or `InMemoryDataset` class and implement the necessary functions (more info [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-your-own-datasets)). The dataset must return torch-geometric `Data` 
objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `dy` (derivative of the label w.r.t atom coordinates) or both.

### Custom prior models
In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be
done by implementing a new prior model class in `torchmdnet.priors` and adding the argument `--prior-model <PriorModelName>`.
As an example, have a look at `torchmdnet.priors.Atomref`.

## Multi-Node Training

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once `torchmd-train` is started on all nodes, a network connection between the nodes will be established using NCCL.

In addition to the environment variables the argument `--num-nodes` has to be specified with the number of nodes involved during training.

```
export NODE_RANK=0
export MASTER_ADDR=hostname1
export MASTER_PORT=12910

mkdir -p output
CUDA_VISIBLE_DEVICES=0,1 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml.yaml --num-nodes 2 --log-dir output/
```

- `NODE_RANK` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- `MASTER_ADDR` : Hostname or IP address of the main node. The same for all involved nodes.
- `MASTER_PORT` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.


### Known Limitations
- Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
- We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).


## Cite
If you use TorchMD-NET in your research, please cite the following papers:

### Main reference
```
@inproceedings{
tholke2021equivariant,
title={Equivariant Transformers for Neural Network based Molecular Potentials},
author={Philipp Th{\"o}lke and Gianni De Fabritiis},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=zNHzqZ9wrRB}
}
```

### Graph Network 

```
@misc{majewski2022machine,
      title={Machine Learning Coarse-Grained Potentials of Protein Thermodynamics}, 
      author={Maciej Majewski and Adrià Pérez and Philipp Thölke and Stefan Doerr and Nicholas E. Charron and Toni Giorgino and Brooke E. Husic and Cecilia Clementi and Frank Noé and Gianni De Fabritiis},
      year={2022},
      eprint={2212.07492},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

### TensorNet

```
@misc{simeon2023tensornet,
      title={TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials}, 
      author={Guillem Simeon and Gianni de Fabritiis},
      year={2023},
      eprint={2306.06482},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
