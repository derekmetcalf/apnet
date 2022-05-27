import os, sys
import apnet
import pickle
import argparse
import numpy as np
from atom_model import AtomModel
from pair_model import PairModel

parser = argparse.ArgumentParser(description='Train single model with provided train and test set')

# Mandatory arguments
parser.add_argument('training_path', help='[string] Path to training pkl file')
parser.add_argument('validation_path', help='[string] Path to validation pkl file')
parser.add_argument('model_name', help='[string] Model name for saving')
parser.add_argument('mode', help='[string] Model mode from [lig, lig-pair, lig-pair-prot]')
parser.add_argument('epochs', help='number of epochs')
parser.add_argument('xfer_path', default=None, help='[string] Path to model to transfer learn from')

args = parser.parse_args(sys.argv[1:])
train_path = args.training_path
val_path = args.validation_path
model_name = args.model_name
mode = args.mode
epochs = int(args.epochs)
xfer_path = args.xfer_path

if __name__ == "__main__":
    dim_t, en_t, _, dim_v, en_v, _ = apnet.util.load_dataset(train_path, val_path)
    if xfer_path is not None:
        pair_model = PairModel(mode=mode).from_file(xfer_path)
    else:
        pair_model = PairModel(mode=mode)
    pair_model.train(dim_t, en_t, dim_v, en_v, f'pair_models/{model_name}', n_epochs=epochs)
