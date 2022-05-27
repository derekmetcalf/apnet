import os, sys

import apnet
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Train single model with provided train and test set')

# Mandatory arguments
parser.add_argument('validation_path', help='[string] Path to validation pkl file')
parser.add_argument('model_path', help='[string] Path of saved model')
parser.add_argument('mode', help='[string] Model mode from [lig, lig-pair, lig-pair-prot]')

args = parser.parse_args(sys.argv[1:])
val_path = args.validation_path
model_path = args.model_path
mode = args.mode

if __name__ == "__main__":

    preds, labs, ligpred, pairpred, source, target = apnet.util.test_dataset(set_name=val_path, model_path=model_path)

    ens_preds = np.squeeze(np.array(preds))
    labs = np.array(labs)
    avg_preds = np.mean(ens_preds, axis=0)
    pred_unc = np.std(ens_preds, axis=0)
    np.save(f'{model_path}_test_preds.npy', ens_preds)
    np.save(f'{model_path}_test_labs.npy', labs)
    np.save(f'{model_path}_lig_preds.npy', ligpred)
    np.save(f'{model_path}_pair_preds.npy', pairpred)
    np.save(f'{model_path}_sources.npy', source)
    np.save(f'{model_path}_targets.npy', target)
    
    print(f'Predictions on {val_path} saved to {model_path}_test_preds.npy')
