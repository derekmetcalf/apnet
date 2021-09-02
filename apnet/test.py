import apnet
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



if __name__ == "__main__":
   
    set_name = "pdbbind"
    mode = "lig-pair"
    suffix = 'xval5'
    ens_preds = []
    for i in range(2):
        model_path = f"pair_models/{set_name}_{mode}_{suffix}_fold{i}"
        preds, labs = apnet.util.test_dataset(set_name, model_path)
        ens_preds.append(preds)
    ens_preds = np.squeeze(np.array(ens_preds))
    labs = np.array(labs)
    avg_preds = np.mean(ens_preds, axis=0)
    pred_unc = np.std(ens_preds, axis=0)
    np.save(f'{set_name}_{mode}_{suffix}_preds.npy', ens_preds)
    np.save(f'{set_name}_{mode}_{suffix}_labs.npy', labs)
    
    print(ens_preds)
    print(labs)
    print(ens_preds.shape)
    print(labs.shape)

