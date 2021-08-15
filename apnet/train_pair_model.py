import apnet
import pandas as pd
import numpy as np
from atom_model import AtomModel
from pair_model import PairModel

#dim_t, en_t = apnet.util.load_dG_dataset('data/pdbbind_multi_smol.pkl', poses=4)
#dim_v, en_v = apnet.util.load_dG_dataset('data/pdbbind_multi_smol.pkl', poses=4)
dim_t, en_t = apnet.util.load_dG_dataset('data/pdbbind_multi_train.pkl', poses=3)
dim_v, en_v = apnet.util.load_dG_dataset('data/pdbbind_multi_val.pkl', poses=3)

atom_model = AtomModel().from_file('atom_models/atom_model2')

pair_model = PairModel(atom_model=atom_model, multipose=True)
#pair_model = PairModel(multipose=True)
pair_model.train(dim_t, en_t, dim_v, en_v, 'pair_models/pdbbind_3pose_attention_att1', n_epochs=1000)

