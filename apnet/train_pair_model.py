import os, sys
import argparse
import apnet
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Single-model training of binding free energy neural networks')

parser.add_argument('--set', type=str, default='', help='Pre-programmed dataset name to train/validate on')
parser.add_argument('--mode', type=str, default='lig-pair', choices=['lig','lig-pair'], help='Whether to train the model on just ligand geometry (lig), or the ligand and its relation to the protein (lig-pair)')
parser.add_argument('--suffix', type=str, default='', help='Saved model name suffix if desired')
parser.add_argument('--delta_path',  default=None, help='Location of base model from which to delta learn (WIP)')
parser.add_argument('--xfer_path', default=None, help='Location of base model from which to transfer learn (WIP)')
parser.add_argument('--epochs', type=str, default='1000', help='Number of epochs to train for')
parser.add_argument('--att', dest='att', action='store_true', help='Train with atom-pair attention mechanism')
parser.add_argument('--no-att', dest='att', action='store_false', help='Train without attention mechanism')
parser.set_defaults(att=False)
parser.add_argument('--pretrained_atom', type=str, default='False', help='Whether to use pre-trained atomic featurization from existing model')
parser.add_argument('--pretrained-atom', dest='pretrained_atom', action='store_true', help='Train with pre-trained atomic features')
parser.add_argument('--no-pretrained-atom', dest='pretrained_atom', action='store_false', help='Train without pre-trained atomic features')
parser.set_defaults(pretrained_atom=False)
parser.add_argument('--lr', type=str, default='5e-4', help='Maximum learning rate (currently fixed 5epoch warmup and cosine decay)')
parser.add_argument('--mp', dest='mp', action='store_true', help='Allow message-passing steps while training the model')
parser.add_argument('--no-mp', dest='mp', action='store_false', help='Disallow message-passing steps while training the model')
parser.set_defaults(mp=False)
parser.add_argument('--dropout', type=float, default=0.2, help='Fraction of dropout to include on all dense layers')
parser.add_argument('--online_aug', type=float, default=0.0, help='Whether to use online data augmentation (in this case, small Cartesian noise injection)')

args = parser.parse_args()
set_name = args.set
mode = args.mode
suffix = args.suffix
delta_path = args.delta_path
xfer_path = args.xfer_path
n_epochs = int(args.epochs)
attention = args.att
pretrained_atom = args.pretrained_atom
lr = float(args.lr)
message_passing = args.mp
dropout = float(args.dropout)
online_aug = float(args.online_aug)

if __name__ == "__main__":
   
    #xfer_path = 'pair_models/pdbbind_pdbbind_pair_prot_lig_readout_dropout_1'
    #xfer_path = None
    #set_name = "pdbbind-xval"
    # mode options are: 'lig', 'lig-pair', 'prot-lig-pair'
    #mode = "lig"
    #mode = "lig-pair"
    #suffix = "prot-lig-pair-att1-highlr"
    #suffix = "att-1"
    #suffix = "ligonly_mp0"
    #suffix=None
    #delta_path = 'pair_models/4MXO_lig_simple2'
    #delta_path = None
    #n_epochs=1000
    
    apnet.util.train_single(set_name,
                            suffix,
                            n_epochs,
                            delta_base=delta_path,
                            xfer=xfer_path,
                            mode=mode,
                            pretrained_atom=pretrained_atom,
                            val_frac=0.1,
                            attention=attention,
                            lr=lr,
                            message_passing=message_passing,
                            dropout=dropout,
                            online_aug=online_aug)
    #apnet.util.train_crossval(set_name, suffix, n_epochs, delta_base=delta_path, xfer=xfer_path, mode=mode, val_frac=0.1, folds=5)
