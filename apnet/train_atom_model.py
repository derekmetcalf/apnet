import apnet
import pandas as pd
import numpy as np
from atom_model import AtomModel

mol, mult, vol, val = apnet.load_monomer_dataset('/theoryfs2/common/data/monomer-pickles/hfadz.pkl')

atom_model = AtomModel()

mol_t = mol[:35000]
mult_t = mult[:35000]
mol_v = mol[35000:]
mult_v = mult[35000:]
model = atom_model.train(mol_t, mult_t, mol_v, mult_v, 'atom_models/atom_model2', n_epochs=500)


