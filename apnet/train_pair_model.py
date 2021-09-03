import apnet
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



if __name__ == "__main__":
   
    #xfer_path = 'pair_models/pdbbind_pdbbind_pair_prot_lig_readout_dropout_1'
    xfer_path = None
    set_name = "4MXO"
    # mode options are: 'lig', 'lig-pair', 'prot-lig-pair'
    mode = "lig"
    #suffix = "prot-lig-pair-att1-highlr"
    #suffix = "att-1"
    suffix = "example-model5"
    #suffix=None
    #delta_path = 'pair_models/4MXO_lig_simple2'
    delta_path = None

    n_epochs=500
    
    apnet.util.train_single(set_name, suffix, n_epochs, delta_base=delta_path, xfer=xfer_path, mode=mode, val_frac=0.1)
    #apnet.util.train_crossval(set_name, suffix, n_epochs, delta_base=delta_path, xfer=xfer_path, mode=mode, val_frac=0.1, folds=5)
