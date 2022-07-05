import os, sys

import pickle
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

    preds, labs, ligpred, pairpred, source, target, shift, pairscale = apnet.util.test_dataset(val_path=val_path, model_path=model_path)

    ens_preds = np.squeeze(preds)
    avg_preds = np.mean(ens_preds, axis=0)
    pred_unc = np.std(ens_preds, axis=0)
    model_info = {"test_preds" : ens_preds,
                  "test_labs"  : labs,
                  "lig_preds"  : ligpred,
                  "pair_preds" : pairpred,
                  "source"     : source,
                  "target"     : target,
                  "shift"      : shift,
                  "pair_scale" : pairscale,
                  }
    with open(f'{model_path}_pred_info.pkl', 'wb') as outf:
        pickle.dump(model_info, outf)
    outf.close()
    print(f'Predictions and metadata on {val_path} saved to {model_path}_pred_info.pkl')
