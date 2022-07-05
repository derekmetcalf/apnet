# AP-Net-dG

AP-Net-dG is a python package for modeling binding free energies and other properties of ligands and proteins.

## Getting Started

#### Environment

It is strongly recommended that you install this package in some kind of virtual environment (like a conda environment).
Create and activate a new conda environment with the following commands:
```
>>> conda create --name apnet python=3.8
>>> conda activate apnet
```
#### Installation

Next, clone this repository and `cd` into the top level of the repository (the same level as this README).
Run the following command to install the `apnet` package (and dependencies) into your current environment:
```
>>> pip install -e .
```
This will take a few minutes.

Try to run the helper scripts on the example systems; first make an input pickle from an .sdf containing the protein as the first molecule and ligands as successive molecules:
(NOTE: this simple script requires pybel, which is somewhat incompatible with some dependencies of AP-Net-dG, so I use a different env for each)
```
>>> python prepare_pkl_allsdf.py data/sets/505_train/training.sdf
```

Do the same for the `505_val/test.sdf` file then try to train a model:
```
>>> python train.py data/sets/505_train data/sets/505_val example-model lig-pair 500
```
This trains a model for 500 epochs in lig-pair mode (Do `>>> python train.py -h` to see documentation for the arguments of this file.).

Now, `test.py` can be used to infer with a model on a new dataset. Here, we'll use it to re-infer on the validation set and save the outputs to numpy arrays:
```
>>> python test.py data/sets/505_val pair_models/example-model lig-pair
```

You may also want to tune a pretrained model on a new dataset in a transfer learning manner. This is easily achieved by using the `--xfer path` keyword in the `train.py` script:

```
>>> python train.py data/sets/505_train data/sets/505_val xfer-example-model lig-pair 500 --xfer_path pair_models/pdbbind-pretrained-eg
```

`pdbbind-pretrained-eg` is a general protein-ligand model trained on PDBBind, and the above script attempts to transfer learn toward a local dataset containing a single protein and several congeneric ligands.
