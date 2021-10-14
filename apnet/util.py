"""
General utility functions for pre-processing molecules
"""

import os
import time
import apnet
import multiprocessing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
import inspect
import pickle
import subprocess

import numpy as np
import pandas as pd
import qcelemental as qcel

import tensorflow as tf


import logging
tf.get_logger().setLevel(logging.ERROR)

from apnet import constants
from multiprocessing import Pool
from atom_model import AtomModel
from pair_model import PairModel

def qcel_to_dimerdata(dimer):
    """ proper qcel mol to ML-ready numpy arrays """

    # this better be a dimer (not a monomer, trimer, etc.)
    if  len(dimer.fragments) != 2:
        #raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dimer.fragments)}")
        return None

    ZA = dimer.symbols[dimer.fragments[0]]
    ZB = dimer.symbols[dimer.fragments[1]]

    # only some elements allowed
    try:
        ZA = np.array([constants.elem_to_z[za] for za in ZA])
        ZB = np.array([constants.elem_to_z[zb] for zb in ZB])
    except:
        return None

    RA = dimer.geometry[dimer.fragments[0]] * constants.au2ang
    RB = dimer.geometry[dimer.fragments[1]] * constants.au2ang

    nA = len(dimer.fragments[0])
    nB = len(dimer.fragments[1])
    aQA = dimer.fragment_charges[0] / nA
    aQB = dimer.fragment_charges[1] / nB

    return (RA, RB, ZA, ZB, aQA, aQB)

def qcel_to_monomerdata(monomer):
    """ proper qcel mol to ML-ready numpy arrays """

    # this better be a monomer 
    if  len(monomer.fragments) != 1:
        raise AssertionError(f"A monomer must have exactly 1 molecular fragment, found {len(monomer.fragments)}")

    Z = monomer.symbols
    Z = np.array([constants.elem_to_z[z] for z in Z])

    R = monomer.geometry * constants.au2ang

    n = len(monomer.symbols)
    aQ = monomer.molecular_charge / n

    return (R, Z, aQ)

def dimerdata_to_qcel(RA, RB, ZA, ZB, aQA, aQB, ignore_ch_mult=False):
    """ ML-ready numpy arrays to qcel mol """
    nA = RA.shape[0]
    nB = RB.shape[0]

    tQA = int(round(aQA * nA))
    tQB = int(round(aQB * nB))

    assert abs(tQA - aQA * nA) < 1e-6
    assert abs(tQB - aQB * nB) < 1e-6

    if not ignore_ch_mult:
        blockA = f"{tQA} {1}\n"
        for ia in range(nA):
            blockA += f"{constants.z_to_elem[ZA[ia]]} {RA[ia,0]} {RA[ia,1]} {RA[ia,2]}\n"

        blockB = f"{tQB} {1}\n"
        for ib in range(nB):
            blockB += f"{constants.z_to_elem[ZB[ib]]} {RB[ib,0]} {RB[ib,1]} {RB[ib,2]}\n"

        dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
        dimer = qcel.models.Molecule.from_data(dimer)
    else:
        mult_A = (sum(ZA) % 2) + 1
        mult_B = (sum(ZB) % 2) + 1
        blockA = f"{tQA} {mult_A}\n"
        for ia in range(nA):
            if ZA[ia] != 0:
                blockA += f"{constants.z_to_elem[ZA[ia]]} {RA[ia,0]} {RA[ia,1]} {RA[ia,2]}\n"

        blockB = f"{tQB} {mult_B}\n"
        for ib in range(nB):
            if ZB[ib] != 0:
                blockB += f"{constants.z_to_elem[ZB[ib]]} {RB[ib,0]} {RB[ib,1]} {RB[ib,2]}\n"
        #mon_test = qcel.models.Molecule.from_data(blockB)

        dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
        try:
            dimer = qcel.models.Molecule.from_data(dimer)
        except:
            dimer = None
    return dimer

def monomerdata_to_qcel(R, Z, aQ):
    """ ML-ready numpy arrays to qcel mol """

    n = R.shape[0]

    tQ = int(round(aQ * n))

    assert abs(tQ - aQ * n) < 1e-6

    block = f"{tQ} {1}\n"
    for i in range(n):
        block += f"{constants.z_to_elem[Z[i]]} {R[i,0]} {R[i,1]} {R[i,2]}\n"

    monomer = block + "no_com\nno_reorient\nunits angstrom"
    monomer = qcel.models.Molecule.from_data(monomer)
    return monomer

def load_bms_dimer(file):
    """Load a single dimer from the BMS-xyz format

    This function expects an xyz file in the format used with the 1.66M dimer dataset. 
    The first line contains the number of atoms. 
    The second line contains a comma-separated list of values such as the dimer name, dimer and monomer charges, SAPT labels (at various levels of theory), and number of atoms in the first monomer.
    The next `natom` lines each contain an atomic symbol follwed by the x, y, and z cooordinates of the atom (Angstrom)

    Parameters
    ----------
    file : str
        The name of a file containing the xyz
    
    Returns
    -------
    dimer : :class:`~qcelemental.models.Molecule`
    labels : :class:`~numpy.ndarray`
        The SAPT0/aug-cc-pV(D+d)Z interaction energy labels: [total, electrostatics, exchange, induction, and dispersion].
    """

    lines = open(file, 'r').readlines()

    natom = int(lines[0].strip())
    dimerinfo = (''.join(lines[1:-natom])).split(',')
    geom = lines[-natom:]


    nA = int(dimerinfo[-1])
    nB = natom - nA
    TQ = int(dimerinfo[1])
    TQA = int(dimerinfo[2])
    TQB = int(dimerinfo[3])
    assert TQ == (TQA + TQB)

    e_tot_aug = float(dimerinfo[14])
    e_elst_aug = float(dimerinfo[15])
    e_exch_aug = float(dimerinfo[16])
    e_ind_aug = float(dimerinfo[17])
    e_disp_aug = float(dimerinfo[18])

    assert abs(e_tot_aug  - (e_elst_aug + e_exch_aug + e_ind_aug + e_disp_aug)) < 1e-6

    blockA = f"{TQA} 1\n" + "".join(geom[:nA])
    blockB = f"{TQB} 1\n" + "".join(geom[nA:])
    dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
    dimer = qcel.models.Molecule.from_data(dimer)

    label = np.array([e_tot_aug, e_elst_aug, e_exch_aug, e_ind_aug, e_disp_aug])
    return dimer, label

def load_dimer_dataset(file, max_size=None):
    """Load multiple dimers from a :class:`~pandas.DataFrame`

    Loads dimers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`ZA`, `ZB`, `RA`, `RB`, `TQA`, `TQB`] are required.
    `ZA` and `ZB` are atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `RA` and `RB` are atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3.)).
    `TQA` and `TQB` are monomer charges (int).

    The columns [`Total_aug`, `Elst_aug`, `Exch_aug`, `Ind_aug`, and `Disp_aug`] are optional.
    Each column describes SAPT0/aug-cc-pV(D+d)Z labels in kcal / mol (`float`).

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`
    
    Returns
    -------
    dimers : list of :class:`~qcelemental.models.Molecule`
    labels : list of :class:`~numpy.ndarray` or None
        None is returned if SAPT0 label columns are not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    N = len(df.index)

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size

    RA = df.RA.tolist()
    RB = df.RB.tolist()
    ZA = df.ZA.tolist()
    ZB = df.ZB.tolist()
    TQA = df.TQA.tolist()
    TQB = df.TQB.tolist()
    aQA = [TQA[i] / np.sum(ZA[i] > 0) for i in range(N)]
    aQB = [TQB[i] / np.sum(ZB[i] > 0) for i in range(N)]
    try:
        labs = df[['Total_aug', 'Elst_aug', 'Exch_aug', 'Ind_aug', 'Disp_aug']].to_numpy()
    except:
        labs = None

    dimers = []
    labels = []
    for i in range(N):
        dat = dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i])
        if dat:
            dimers.append(dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i]))
            labels.append(labs[i])

    return dimers, labels

def load_dG_dataset(file, poses=None, max_size=None, ignore_ch_mult=True):
    """Load multiple protein-ligand pairs from a :class:`~pandas.DataFrame`

    Loads dimers from the :class:`~pandas.DataFrame` format.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.
    In multipose mode, we may have many such configurations of each molecular dimer.

    The columns [`ZA`, `ZB`, `RA`, `RB`] are required.
    `ZA` and `ZB` are atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `RA` and `RB` are atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3.)).

    The column `label` is optional.
    This column may describe delta G of binding, pKd, or pKi, depending on target (`float`).

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`
    
    Returns
    -------
    dimers : list of :class:`~qcelemental.models.Molecule`
    labels : list of :class:`~numpy.ndarray` or None
        None is returned if SAPT0 label columns are not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    N = len(df.index)

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size

    #RA = [df['RA'].iloc[i] for i in range(len(df))]#df.RA.tolist()
    RA = df.RA.tolist()
    #if len(RA[0][0][0]) == 3 and poses is not None:
    #    RA = [RA[i][0:poses] for i in range(len(RA))]
    RB = df.RB.tolist()
    ZA = df.ZA.tolist()
    ZB = df.ZB.tolist()

    aQA = [0 for i in range(N)]
    aQB = [0 for i in range(N)]
    try:
        all_labels = df['label'].to_numpy(np.float32)
    except:
        all_labels = None

    dimers = []
    labels = []


    #with Pool(multiprocessing.cpu_count()) as p:
    #    input_package = [(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i], all_labels[i]) for i in range(N)]
    #    outs = p.map(multiprocess_qcel, input_package)
    #dimers = [out[0] for out in outs]
    #labels = [out[1] for out in outs]
    for i in range(N):
        #if type(RA[i]) is list:
        configs = []
        any_conf = False
        this_conf = dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i], ignore_ch_mult=ignore_ch_mult)
        #print(this_conf)
        if this_conf:
            configs.append(this_conf)
            labels.append(all_labels[i])

        if len(configs) > 0:
            dimers.append(configs)
            
        #else:
        #    this_dim = dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i], ignore_ch_mult=ignore_ch_mult)
        #    if this_dim:
        #        dimers.append(this_dim)
        #    #dimers.append(dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i], ignore_ch_mult=ignore_ch_mult))
    supp = {}
    normal_cols = ['RA', 'RB', 'ZA', 'ZB', 'label', 'system']
    for column in df.columns:
        if column not in normal_cols:
            supp[column] = df[column].tolist()
    if len(supp) == 0:
        return dimers, labels
    else: return dimers, labels, supp

def trainval_warn(val_frac):
    fn_args = inspect.getfullargspec(load_dataset)
    if val_frac != fn_args.defaults[0]:
        warnings.warn("This dataset has a fixed train/val split, not setting validation fraction.")
    return

def load_dataset(train_path=None, val_path=None, set_name=None, val_frac=0.1):
    binding_db_sets = ["295", "35", "pdbbind-general"]
    dim_t = None
    en_t = None
    dim_v = None
    en_v = None
    if set_name == "pdbbind_multi":
        trainval_warn(val_frac)
        dim_t, en_t = load_dG_dataset('data/pdbbind_multi_train.pkl', poses=1)
        dim_v, en_v = load_dG_dataset('data/pdbbind_multi_val.pkl', poses=1)
    elif set_name == "pdbbind":
        trainval_warn(val_frac)
        dim_t, en_t, supp_t = load_dG_dataset('data/pdbbind_pocket_train.pkl', poses=1)
        dim_v, en_v, supp_v = load_dG_dataset('data/pdbbind_pocket_val.pkl', poses=1)
        return dim_t, en_t, supp_t, dim_v, en_v, supp_t
    elif set_name == "pdbbind_multi_smol":    
        trainval_warn(val_frac)
        dim_t, en_t = load_dG_dataset('data/pdbbind_multi_smol.pkl', poses=1)
        dim_v, en_v = load_dG_dataset('data/pdbbind_multi_smol.pkl', poses=1)
    elif set_name == "4MXO":
        trainval_warn(val_frac)
        dim_t, en_t = load_dG_dataset('data/4MXO_target/4MXO_pocket_train/dimers.pkl', poses=1)
        dim_v, en_v = load_dG_dataset('data/4MXO_target/4MXO_pocket_val/dimers.pkl', poses=1)
    elif set_name == "505":
        trainval_warn(val_frac)
        dim_t, en_t = load_dG_dataset('data/sets/505_train/dimers.pkl', poses=1)
        dim_v, en_v = load_dG_dataset('data/sets/505_val/dimers.pkl', poses=1)

    elif set_name in ['pdbbind-0', 'pdbbind-1', 'pdbbind-2', 'pdbbind-3', 'pdbbind-4']:
        val_fold = int(set_name[-1])
        train_folds = [0, 1, 2, 3, 4]
        train_folds.remove(val_fold)
        dim_v, en_v, supp_v = load_dG_dataset(f'data/pdbbind-xval/fold{val_fold}.pkl')
        dims_t = []
        ens_t = []
        supps_t = []
        for i, fold in enumerate(train_folds):
            dim, en, supp = load_dG_dataset(f'data/pdbbind-xval/fold{fold}.pkl')
            dims_t.extend(dim)
            ens_t.extend(en)
            supps_t.extend(supp)
        #print(dims_t)
        #print(ens_t)
        #print(supps_t)
        #exit()
        dim_t = dims_t
        en_t = ens_t
        supp_t = supps_t
        #dim_t = pd.concat(dims_t)
        #en_t = np.concat(ens_t)

    elif set_name in binding_db_sets:
        np.random.seed(4202)
        if set_name == "pdbbind-general":
            dim, en = load_dG_dataset(f'data/PDBbind-general-v2020/dimers.pkl', poses=1)
        else:
            dim, en = load_dG_dataset(f'data/sets/set{set_name}/dimers.pkl', poses=1)
        shuffler = np.random.permutation(len(en))
        dim_shuff = [dim[i] for i in shuffler]
        en_shuff = [en[i] for i in shuffler]
        train_len = np.ceil(float(len(en_shuff)) * (1 - val_frac))
        dim_t = []
        en_t = []
        dim_v = []
        en_v = []
        for i in range(len(en_shuff)):
            if i <= train_len:
                dim_t.append(dim_shuff[i])
                en_t.append(en_shuff[i])
            else:
                dim_v.append(dim_shuff[i])
                en_v.append(en_shuff[i])

    if train_path is not None:
        dim_t, en_t = load_dG_dataset(os.path.join(train_path, 'dimers.pkl'), poses=1)
    if val_path is not None:
        dim_v, en_v = load_dG_dataset(os.path.join(val_path, 'dimers.pkl'), poses=1)
    return dim_t, en_t, None, dim_v, en_v, None

def test_dataset(model_path=None, val_path=None, set_name=None):
    dim_t, en_t, supp_t, dim_v, en_v, supp_v = load_dataset(set_name=set_name)
    #atom_model = AtomModel().from_file('atom_models/atom_model2')
    pair_model = PairModel().from_file(model_path)
    print("\nProcessing Dataset...", flush=True)
    time_loaddata_start = time.time()
    Nv = len(dim_v)
    inds_v = np.arange(Nv)
    #data_loader_t = apnet.pair_model.PairDataLoader(dim_t, en_t, 5.0, r_cut_im)
    data_loader_v = apnet.pair_model.PairDataLoader(dim_v, en_v, 5.0, 5.0)
    dt_loaddata = time.time() - time_loaddata_start
    print(f"...Done in {dt_loaddata:.2f} seconds", flush=True)
    preds_v = []
    lig_pred_v = []
    pair_pred_v = []
    source_v = []
    target_v = []
    energy_v = en_v
    for inds in inds_v:
        inp_v_chunk = data_loader_v.get_data([inds])
        outs_v = pair_model.model(inp_v_chunk[0])
        preds_v.append(outs_v[0])
        lig_pred_v.append(outs_v[1])
        pair_pred_v.append(outs_v[2])
        source_v.append(outs_v[3])
        target_v.append(outs_v[4])

    preds = np.array(preds_v)
    labs = np.array(energy_v)
    lig_preds = np.array(lig_pred_v)
    pair_preds = np.array(pair_pred_v)
    sources = np.array(source_v)
    targets = np.array(target_v)

    return preds, labs, lig_preds, pair_preds, sources, targets

def train_single(set_name, modelsuffix=None, epochs=1000, delta_base=None, xfer=None, pretrained_atom=False, mode='lig', val_frac=0.2, attention=False, lr=0.0001, message_passing=False, dropout=0.2, online_aug=False):
    dim_t, en_t, supp_t, dim_v, en_v, supp_v = load_dataset(set_name=set_name, val_frac=val_frac)
    dock_vars = ["Prime energy", "Docking score", "MMGBSA dG Bind"]
    ext_t = []
    ext_v = []
    if supp_t:
        for key, value in supp_t.items():
            if key in dock_vars:
                ext_t.append(value)
                ext_v.append(supp_v[key])
    naive_model = np.mean(en_t) - en_v
    print(f'Mean training label        : {np.mean(en_t)}')
    print(f'Mean validation label      : {np.mean(en_v)}')
    print(f'Naive validation RMSE      : {np.sqrt(np.mean(np.square(naive_model)))}')

    if pretrained_atom:
        atom_model = AtomModel().from_file('atom_models/atom_model2')
    else:
        atom_model = None
    if xfer is not None:
        pair_model = PairModel(atom_model=atom_model, mode=mode, attention=attention, message_passing=message_passing, dropout=dropout).from_file(xfer_path)
    elif delta_base is not None:
        delta_base_model = PairModel(atom_model=atom_model, mode='lig').from_file(delta_base)
        #base_val_preds = []
        #for dim in dim_v:
        #    base_val_preds.append(delta_base_model.model(dim, training=False))
        #base_errs = np.array(base_val_preds) - en_v
        #print(f'Base model validation RMSE : {np.sqrt(np.mean(np.square(base_errs)))}')
        pair_model = PairModel(atom_model=atom_model, delta_base=delta_base_model, mode=mode, attention=attention, message_passing=message_passing, dropout=dropout)
    else:
        pair_model = PairModel(atom_model=atom_model, mode=mode, attention=attention, message_passing=message_passing, dropout=dropout)
    

    
    if modelsuffix is not None:
        modelname = f'{set_name}_{mode}_{modelsuffix}'
        pair_model.train(dim_t, en_t, dim_v, en_v, f'pair_models/{modelname}', n_epochs=epochs, ext_t=ext_t, ext_v=ext_v, learning_rate=lr, online_aug=online_aug, log_path=f'pair_models/{modelname}.log')
    else:
        pair_model.train(dim_t, en_t, dim_v, en_v, n_epochs=epochs, ext_t=ext_t, ext_v=ext_v, learning_rate=lr, online_aug=online_aug)
   
    return pair_model

def get_folds(X, y, folds):
    sz = np.ceil(float(len(X)) / float(folds))
    Xs = []
    ys = []
    for i in range(folds):
        start = int(i*sz)
        end = int((i+1)*sz)
        Xs.append(X[start:end])
        ys.append(y[start:end])
    return Xs, ys

def train_crossval(set_name, modelsuffix, epochs, delta_base=None, xfer=None, mode='lig', val_frac=0.2, folds=5):
    DIM_t, EN_t, SUPP_t, DIM_v, EN_v, SUPP_v = load_dataset(set_name, val_frac)
    dim_ts, en_ts = get_folds(DIM_t, EN_t, folds)

    models = []
    for fold in range(folds):
        #atom_model = AtomModel().from_file('atom_models/atom_model2')
        #pair_model = PairModel(atom_model=atom_model, mode=mode)
        t_dim_folds = [dim_ts[x] for x in range(folds) if x != fold]
        t_en_folds = [en_ts[x] for x in range(folds) if x != fold]
        dim_t = []
        en_t = []
        for i, dim_fold in enumerate(t_dim_folds):
            for j, dim in enumerate(dim_fold):
                dim_t.append(dim)
                en_t.append(t_en_folds[i][j])
        dim_v = dim_ts[fold]
        en_v = en_ts[fold]
        pickle.dump((dim_t, en_t), open(f"tmp/{fold}train.pkl", "wb"))
        pickle.dump((dim_v, en_v), open(f"tmp/{fold}val.pkl", "wb"))
        if modelsuffix is not None:
            modelname = f'{set_name}_{mode}_{modelsuffix}_fold{fold}'

        f = open(f"pair_models/{modelname}.out", "w")
        print(f"\n\nStarting fold {fold}")
        subprocess.call(f"python train.py tmp/{fold}train.pkl tmp/{fold}val.pkl {modelname} {mode} {epochs}", shell=True, stdout=f)
        #    pair_model.train(dim_t, en_t, dim_v, en_v, f'pair_models/{modelname}_fold{fold}', n_epochs=epochs)
        #else:
        #    pair_model.train(dim_t, en_t, dim_v, en_v, n_epochs=epochs)
    return

    

def multiprocess_qcel(inp_package):
    RA = inp_package[0]
    RB = inp_package[1]
    ZA = inp_package[2]
    ZB = inp_package[3]
    aQA = inp_package[4]
    aQB = inp_package[5]
    label = inp_package[6]
    configs = []
    for j in range(len(RA)):
        configs.append(dimerdata_to_qcel(RA[j], RB, ZA, ZB, aQA, aQB, ignore_ch_mult=True))
    return (configs, label)

def load_monomer_dataset(file, max_size=None):
    """Load multiple monomers from a :class:`~pandas.DataFrame`

    Loads monomers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`Z`, `R`, and `total_charge`] are required.
    `Z` is atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `R` is atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3)).
    `total_charge` are monomer charges (int).

    The columns [`cartesian_multipoles`, `volume_ratios`, and `valence_widths`] are optional.
    `cartesian_multipoles` describes atom-centered charges, dipoles, and quadrupoles (:class:`~numpy.ndarray` of `float` with shape (`n`, 10). The ordering convention is [q, u_x, u_y, u_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz], all in a.u.)
    `volume_ratios` is the ratio of the volume of the atom-in-molecule to the free atom (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), unitless
    `valence_widths` is the width describing the valence electron density (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), TODO: check units. a.u. ? inverse width?

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`
    
    Returns
    -------
    monomers : list of :class:`~qcelemental.models.Molecule`
    cartesian_multipoles : list of :class:`~numpy.ndarray` or None
        None is returned if the `cartesian_multipoles` column is not present in the :class:`~pandas.DataFrame`
    volume_ratios : list of :class:`~numpy.ndarray` or None
        None is returned if the `volume_ratios` column is not present in the :class:`~pandas.DataFrame`
    valence_widths : list of :class:`~numpy.ndarray` or None
        None is returned if the `valence_widths` column is not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    N = len(df.index)

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size

    R = df.R.tolist()
    Z = df.Z.tolist()
    TQ = df.total_charge.tolist()
    aQ = [TQ[i] / np.sum(Z[i] > 0) for i in range(N)]

    try:
        cartesian_multipoles = df['cartesian_multipoles'].to_numpy()
    except:
        cartesian_multipoles = None

    try:
        volume_ratios = df['volume_ratios'].to_numpy()
    except:
        volume_ratios = None

    try:
        valence_widths = df['valence_widths'].to_numpy()
    except:
        valence_widths = None

    monomers = []
    for i in range(N):
        monomers.append(monomerdata_to_qcel(R[i], Z[i], aQ[i]))

    return monomers, cartesian_multipoles, volume_ratios, valence_widths



if __name__ == "__main__":

    mol = qcel.models.Molecule.from_data("""
    0 1
    O 0.000000 0.000000 0.100000
    H 1.000000 0.000000 0.000000
    CL 0.000000 1.000000 0.400000
    --
    0 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """)
    print(mol.to_string("psi4"))
    print(mol)

    data = qcel_to_dimerdata(mol)

    mol2 = dimerdata_to_qcel(*data)
    print(mol2.to_string("psi4"))
    print(mol2)


    mol3 = qcel.models.Molecule.from_data("""
    -2 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """)

    R, Z, aQ = qcel_to_monomerdata(mol3)
    print(mol3)
    print(R)
    print(Z)
    print(aQ)


    #dimers, labels = load_dimer_dataset("data/200_dimers.pkl")
    #load_dimer_dataset("/theoryfs2/common/data/dimer-pickles/1600K_val_dimers-fixed.pkl")
    #print(labels)
