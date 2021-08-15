import sys, os
import multiprocessing
import math
import time
import numpy as np

from pathlib import Path
ROOT_DIR = Path(__file__).parent

import tensorflow as tf

from apnet.keras_pair_model import KerasPairModel
from apnet import constants
from multiprocessing import Pool, get_context

class PairDataLoader:
    """ todo """

    def __init__(self, dimers, energies, r_cut, r_cut_im, multipose=False):

        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.N = len(dimers)
        self.multipose = multipose
        if energies is not None:
            print(len(energies))
            print(self.N)
            assert len(energies) == self.N

        self.RA_list = []
        self.RB_list = []
        self.ZA_list = []
        self.ZB_list = []
        self.total_charge_A_list = []
        self.total_charge_B_list = []

        self.e_AA_source_list = []
        self.e_AA_target_list = []

        self.e_BB_source_list = []
        self.e_BB_target_list = []

        self.e_ABsr_source_list = []
        self.e_ABsr_target_list = []
        #self.e_ABlr_source_list = []
        #self.e_ABlr_target_list = []

        #with get_context("spawn").Pool() as p:#multiprocessing.cpu_count()) as p:
        parsed_data = []
        for dimer in dimers:
            parsed_data.append(self.multiprocess_loader(dimer))
        self.RA_list = [parsed_datum[0] for parsed_datum in parsed_data]
        self.RB_list = [parsed_datum[1] for parsed_datum in parsed_data]
        self.ZA_list = [parsed_datum[2] for parsed_datum in parsed_data]
        self.ZB_list = [parsed_datum[3] for parsed_datum in parsed_data]
        self.total_charge_A_list = [parsed_datum[4] for parsed_datum in parsed_data]
        self.total_charge_B_list = [parsed_datum[5] for parsed_datum in parsed_data]

        self.e_AA_source_list = [parsed_datum[6] for parsed_datum in parsed_data]
        self.e_AA_target_list = [parsed_datum[7] for parsed_datum in parsed_data]

        self.e_BB_source_list = [parsed_datum[8] for parsed_datum in parsed_data]
        self.e_BB_target_list = [parsed_datum[9] for parsed_datum in parsed_data]

        self.e_ABsr_source_list = [parsed_datum[10] for parsed_datum in parsed_data]
        self.e_ABsr_target_list = [parsed_datum[11] for parsed_datum in parsed_data]

        #for ind in range(self.N):
        #    
        #    dimer = dimers[ind]

        #    RA, RB, ZA, ZB, total_charge_A, total_charge_B = self.dimer_to_data(dimer)

        #    e_AA_sources = []
        #    e_AA_targets = []
        #    e_BB_sources = []
        #    e_BB_targets = []
        #    e_ABsr_sources = []
        #    e_ABsr_targets = []
        #    #e_ABlr_sources = []
        #    #e_ABlr_targets = []

        #    for pose_ind in range(len(RA)):
        #        e_AA_source, e_AA_target = self.edges(RA[pose_ind])
        #        e_BB_source, e_BB_target = self.edges(RB[pose_ind])
        #        e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = self.edges_im(RA[pose_ind], RB[pose_ind])
        #        e_AA_sources.append(e_AA_source)
        #        e_AA_targets.append(e_AA_target)
        #        e_BB_sources.append(e_BB_source)
        #        e_BB_targets.append(e_BB_target)
        #        e_ABsr_sources.append(e_ABsr_source)
        #        e_ABsr_targets.append(e_ABsr_target)
        #        #e_ABlr_sources.append(e_ABlr_source)
        #        #e_ABlr_targets.append(e_ABlr_target)

        #    self.RA_list.append(RA)
        #    self.RB_list.append(RB)
        #    self.ZA_list.append(ZA)
        #    self.ZB_list.append(ZB)
        #    self.total_charge_A_list.append(total_charge_A)
        #    self.total_charge_B_list.append(total_charge_B)

        #    self.e_AA_source_list.append(e_AA_sources)
        #    self.e_AA_target_list.append(e_AA_targets)

        #    self.e_BB_source_list.append(e_BB_sources)
        #    self.e_BB_target_list.append(e_BB_targets)

        #    self.e_ABsr_source_list.append(e_ABsr_sources)
        #    self.e_ABsr_target_list.append(e_ABsr_targets)
        #    #self.e_ABlr_source_list.append(e_ABlr_sources)
        #    #self.e_ABlr_target_list.append(e_ABlr_targets)

        self.IE_list = []

        if energies is not None:
            self.has_energies = True
            self.IE_list = np.array(energies, dtype=np.float32)
            # todo: clean this up
            if len(self.IE_list.shape) > 1:
                if self.IE_list.shape[1] == 4:
                    pass
                elif self.IE_list.shape[1] == 5:
                    self.IE_list = self.IE_list[:,1:]
            else:
                pass
        else:
            self.has_energies = False

    def multiprocess_loader(self, dimer):
        RA, RB, ZA, ZB, total_charge_A, total_charge_B = self.dimer_to_data(dimer)
    
        e_AA_sources = []
        e_AA_targets = []
        e_BB_sources = []
        e_BB_targets = []
        e_ABsr_sources = []
        e_ABsr_targets = []
        #e_ABlr_sources = []
        #e_ABlr_targets = []
    
        for pose_ind in range(len(RA)):
            e_AA_source, e_AA_target = self.edges(RA[pose_ind])
            e_BB_source, e_BB_target = self.edges(RB[pose_ind])
            e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = self.edges_im(RA[pose_ind], RB[pose_ind])
            e_AA_sources.append(e_AA_source)
            e_AA_targets.append(e_AA_target)
            e_BB_sources.append(e_BB_source)
            e_BB_targets.append(e_BB_target)
            e_ABsr_sources.append(e_ABsr_source)
            e_ABsr_targets.append(e_ABsr_target)
            #e_ABlr_sources.append(e_ABlr_source)
            #e_ABlr_targets.append(e_ABlr_target)
    
        return (RA, RB, ZA, ZB, total_charge_A, total_charge_B, e_AA_sources, e_AA_targets, e_BB_sources, e_BB_targets, e_ABsr_sources, e_ABsr_targets)

    def get_data(self, inds):

        inp = { 'RA' : [],              # N_sys X N_poses X N_atoms_A X 3
                'RB' : [],              # N_sys X N_poses X N_atoms_B X 3
                'ZA' : [],              # N_sys X N_poses X N_atoms_A
                'ZB' : [],              # N_sys X N_poses X N_atoms_B
                'e_ABsr_source' : [],   # N_sys X N_poses X N_sr_neighborsAB 
                'e_ABsr_target' : [],   # N_sys X N_poses X N_sr_neighborsAB   
                #'e_ABlr_source' : [],   # N_sys X N_poses X N_lr_neighborsAB 
                #'e_ABlr_target' : [],   # N_sys X N_poses X N_lr_neighborsAB 
                'e_AA_source' : [],     # N_sys X N_poses X N_sr_neighborsAA 
                'e_AA_target' : [],     # N_sys X N_poses X N_sr_neighborsAA 
                'e_BB_source' : [],     # N_sys X N_poses X N_sr_neighborsBB 
                'e_BB_target' : [],     # N_sys X N_poses X N_sr_neighborsBB 
                'dimer_ind' : [],       # N_sys X N_poses X N_sr_neighborsAB
                #'dimer_ind_lr' : [],    # N_sys X N_poses X N_lr_neighborsAB
                'monomerA_ind' : [],    # N_sys X N_poses X N_atoms_A 
                'monomerB_ind' : [],    # N_sys X N_poses X N_atoms_B
                'total_charge_A' : [],  # N_sys X N_poses
                'total_charge_B' : [],  # N_sys X N_poses
              }

        offsetA, offsetB = 0, 0
        for i, ind in enumerate(inds): # terrible enumeration variable names
            inp['RA'].append(self.RA_list[ind])
            inp['ZA'].append(self.ZA_list[ind])
            inp['RB'].append(self.RB_list[ind])
            inp['ZB'].append(self.ZB_list[ind])
            inp['total_charge_A'].append([self.total_charge_A_list[ind]])
            inp['total_charge_B'].append([self.total_charge_B_list[ind]])
            inp['e_ABsr_source'].append([sources + offsetA for sources in self.e_ABsr_source_list[ind]])
            inp['e_ABsr_target'].append([targets + offsetB for targets in self.e_ABsr_target_list[ind]])
            #inp['e_ABlr_source'].append([sources + offsetA for sources in self.e_ABlr_source_list[ind]])
            #inp['e_ABlr_target'].append([targets + offsetB for targets in self.e_ABlr_target_list[ind]])
            inp['e_AA_source'].append([sources + offsetA for sources in self.e_AA_source_list[ind]])
            inp['e_AA_target'].append([targets + offsetA for targets in self.e_AA_target_list[ind]])
            inp['e_BB_source'].append([sources + offsetA for sources in self.e_BB_source_list[ind]])
            inp['e_BB_target'].append([targets + offsetA for targets in self.e_BB_target_list[ind]])
            inp['dimer_ind'].append([np.full(len(src_len), i*len(self.e_ABsr_source_list[ind])+j) for j, src_len in enumerate(self.e_ABsr_source_list[ind])])
            #inp['dimer_ind_lr'].append([np.full(len(src_len), i*len(self.e_ABsr_source_list[ind])+j) for j, src_len in enumerate(self.e_ABlr_source_list[ind])])
            inp['monomerA_ind'].append([np.full(len(R_len), i*len(self.RA_list[ind])+j) for j, R_len in enumerate(self.RA_list[ind])])
            inp['monomerB_ind'].append([np.full(len(R_len), i*len(self.RB_list[ind])+j) for j, R_len in enumerate(self.RB_list[ind])])
            offsetA += self.RA_list[ind][0].shape[0] # [ind][0] hardcode implies all RA and RB in a multipose
            offsetB += self.RB_list[ind][0].shape[0] #   have the same number of atoms

        #print(np.array(self.RA_list[0]).shape)
        #print(np.array(self.RB_list[0]).shape)
        #print(np.array(inp['RA'][0]).shape)
        #print(np.array(inp['RB'][0]).shape)
        #print(np.array(inp['dimer_ind']))
        #print(np.array(inp['dimer_ind']).shape)
        #print(np.array(inp['dimer_ind'][0]).shape)
        #exit()

        ### For now, opting to do the following to retain flexibility in handling disparate numbers
        ### of poses between examples.
        if not self.multipose:
            for k, v in inp.items():
                inp[k] = [np.concatenate(v[sys], axis=0) for sys in range(len(v))]
        else:
            #print(len(inp['RA']))
            for k, v in inp.items():
                inp[k] = np.concatenate(v[0], axis=0)
                #inp[k] = np.squeeze(np.concatenate(v[0], axis=0))
            #print(len(inp['RA']))
        ###

        if not self.has_energies:
            return inp

        target_ie = np.array([self.IE_list[ind] for ind in inds])

        return inp, target_ie

    def edges(self, R):
    
        natom = np.shape(R)[0]
    
        RA = np.expand_dims(R, 0)
        RB = np.expand_dims(R, 1)
    
        RA = np.tile(RA, [natom,1,1])
        RB = np.tile(RB, [1,natom,1])
    
        dist = np.linalg.norm(RA - RB, axis=2)
    
        mask = np.logical_and(dist < self.r_cut, dist > 0.0)
        edges = np.where(mask) # dimensions [n_edge x 2]
    
        return edges[0], edges[1]


    def edges_im(self, RA, RB):
    
        natomA = tf.shape(RA)[0]
        natomB = tf.shape(RB)[0]
    
        RA_temp = tf.expand_dims(RA, 1)
        RB_temp = tf.expand_dims(RB, 0)

        RA_temp = tf.tile(RA_temp, [1, natomB, 1])
        RB_temp = tf.tile(RB_temp, [natomA, 1, 1])
    
        dist = tf.norm(RA_temp - RB_temp, axis=2)
    
        mask = (dist <= self.r_cut_im)
        edges_sr = tf.where(mask) # dimensions [n_edge x 2]
        edges_lr = tf.where(tf.math.logical_not(mask)) # dimensions [n_edge x 2]
    
        return edges_sr[:,0], edges_sr[:,1], edges_lr[:,0], edges_lr[:,1]

    def make_quad(self, flat_quad):
    
        natom = flat_quad.shape[0]
        full_quad = np.zeros((natom, 3, 3))
        full_quad[:,0,0] = flat_quad[:,0] # xx
        full_quad[:,0,1] = flat_quad[:,1] # xy
        full_quad[:,1,0] = flat_quad[:,1] # xy
        full_quad[:,0,2] = flat_quad[:,2] # xz
        full_quad[:,2,0] = flat_quad[:,2] # xz
        full_quad[:,1,1] = flat_quad[:,3] # yy
        full_quad[:,1,2] = flat_quad[:,4] # yz
        full_quad[:,2,1] = flat_quad[:,4] # yz
        full_quad[:,2,2] = flat_quad[:,5] # zz
    
        trace = full_quad[:,0,0] + full_quad[:,1,1] + full_quad[:,2,2]
    
        full_quad[:,0,0] -= trace / 3.0
        full_quad[:,1,1] -= trace / 3.0
        full_quad[:,2,2] -= trace / 3.0
    
        return full_quad


    def dimer_to_data(self, dimer):
        """ QCelemental molecule to ML-ready numpy arrays """

        if type(dimer) is list:
            multipose = True
            RAs = []
            RBs = []
            ZAs = []
            ZBs = []
            total_charges_A = []
            total_charges_B = []
            for dim in dimer:
                if len(dim.fragments) != 2:
                    raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dim.fragments)}")
                    return None
                RAs.append(np.array(dim.geometry[dim.fragments[0]], dtype=np.float32) * constants.au2ang)
                RBs.append(np.array(dim.geometry[dim.fragments[1]], dtype=np.float32) * constants.au2ang)
                ZAs.append(np.array([constants.elem_to_z[za] for za in dim.symbols[dim.fragments[0]]], dtype=np.float32))
                ZBs.append(np.array([constants.elem_to_z[zb] for zb in dim.symbols[dim.fragments[1]]], dtype=np.float32))
                total_charges_A.append(int(dim.fragment_charges[0]))
                total_charges_B.append(int(dim.fragment_charges[1]))
            return (RAs, RBs, ZAs, ZBs, total_charges_A, total_charges_B)

        else:
            # this better be a dimer (not a monomer, trimer, etc.)
            if  len(dimer.fragments) != 2:
                raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dimer.fragments)}")
                return None

            RA = np.array(dimer.geometry[dimer.fragments[0]], dtype=np.float32) * constants.au2ang
            RB = np.array(dimer.geometry[dimer.fragments[1]], dtype=np.float32) * constants.au2ang

            # only some elements allowed; todo: better error message
            try:
                # todo: int
                ZA = np.array([constants.elem_to_z[za] for za in dimer.symbols[dimer.fragments[0]]], dtype=np.float32)
                ZB = np.array([constants.elem_to_z[zb] for zb in dimer.symbols[dimer.fragments[1]]], dtype=np.float32)
            except:
                return None

            total_charge_A = int(dimer.fragment_charges[0])
            total_charge_B = int(dimer.fragment_charges[1])
            return ([RA], [RB], [ZA], [ZB], [total_charge_A], [total_charge_B])
        

class PairModel:
    """ todo """

    def __init__(self, atom_model=None, multipose=False, **kwargs):

        # todo : pass params
        # todo : better atom_model handling
        self.atom_model = atom_model
        if atom_model is not None:
            self.model = KerasPairModel(atom_model.model)
        else:
            self.model = KerasPairModel()
        self.multipose = multipose
        if multipose:
            self.batch_size = 1

    @classmethod
    def from_file(cls, model_path):

        obj = cls(None)
        obj.model = tf.keras.models.load_model(model_path)
        return obj

    @classmethod
    def pretrained(cls, index=0):

        obj = cls(None)
        model_path = f"{ROOT_DIR}/pair_models/pair{index}"
        obj.model = tf.keras.models.load_model(model_path)
        return obj

    def train(self, dimers_t, energies_t, dimers_v, energies_v, model_path=None, log_path=None, **kwargs):

        # redirect stdout to log file, if specified
        if log_path is not None:
            default_stdout = sys.stdout
            log_file = open(log_path, "a")
            sys.stdout = log_file

        # refuse to overwrite an existing model file, if specified
        if model_path is not None:
            if os.path.exists(model_path):
                raise Exception(f"{model_path=} already exists. Delete existing model or choose a new `model_path`")

        print("~~ Training Pair Model ~~", flush=True)
        # todo : print time and date. maybe machine specs?

        if model_path is not None:
            print(f"\nSaving model to '{model_path}'", flush=True)
        else:
            print("\nNo `model_path` provided, not saving model", flush=True)

        # network hyperparameters
        # todo: get kwargs from init(), not from train()
        n_message = kwargs.get("n_message", 3)
        n_neuron = kwargs.get("n_neuron", 128)
        n_embed = kwargs.get("n_embed", 8)
        n_rbf = kwargs.get("n_rbf", 8)
        r_cut_im = kwargs.get("r_cut_im", 8.0)

        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {n_message=}", flush=True)
        print(f"  {n_neuron=}", flush=True)
        print(f"  {n_embed=}", flush=True)
        print(f"  {n_rbf=}", flush=True)
        print(f"  {r_cut_im=}", flush=True)
        
        # training hyperparameters
        n_epochs = kwargs.get("n_epochs", 15)
        batch_size = kwargs.get("batch_size", 16)
        learning_rate = kwargs.get("learning_rate", 0.0010)
        learning_rate_decay = 0.0 #TODO
        if self.multipose:
            batch_size = self.batch_size

        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {batch_size=}", flush=True)
        print(f"  {learning_rate=}", flush=True)
        print(f"  {learning_rate_decay=}", flush=True)

        Nt = len(dimers_t)
        Nv = len(dimers_v)

        print("\nDataset:", flush=True)
        print(f"  n_dimers_train={Nt}", flush=True)
        print(f"  n_dimers_val={Nv}", flush=True)

        inds_t = np.arange(Nt)
        inds_v = np.arange(Nv)

        np.random.seed(4201)
        np.random.shuffle(inds_t)
        num_batches = math.ceil(Nt / batch_size)

        # TODO: replaced hardcoded 200 dimers. Probably want a data_loader.get_large_batch
        #inds_t_chunks = [inds_t[i*200:min((i+1)*200,Nt)] for i in range(math.ceil(Nt / 200))]
        #inds_v_chunks = [inds_v[i*200:min((i+1)*200,Nv)] for i in range(math.ceil(Nv / 200))]

        print("\nProcessing Dataset...", flush=True)
        time_loaddata_start = time.time()
        data_loader_t = PairDataLoader(dimers_t, energies_t, 5.0, r_cut_im, self.multipose)
        data_loader_v = PairDataLoader(dimers_v, energies_v, 5.0, r_cut_im, self.multipose)
        dt_loaddata = time.time() - time_loaddata_start
        print(f"...Done in {dt_loaddata:.2f} seconds", flush=True)

        #inp_t, energy_t = data_loader_t.get_data(inds_t)
        #inp_v, energy_v = data_loader_v.get_data(inds_v)

        #inp_t_chunks = [data_loader_t.get_data(inds_t_i) for inds_t_i in inds_t_chunks]
        #inp_v_chunks = [data_loader_v.get_data(inds_v_i) for inds_v_i in inds_v_chunks]
        preds_t = []
        preds_v = []
        energy_t = []
        energy_v = []
        for inds in inds_t:
            inp_t_chunk = data_loader_t.get_data([inds])
            preds_t.append(np.sum(test_batch(self.model, inp_t_chunk[0])))
            energy_t.append(inp_t_chunk[1][0])
        for inds in inds_v:
            inp_v_chunk = data_loader_v.get_data([inds])
            preds_v.append(np.sum(test_batch(self.model, inp_v_chunk[0])))
            energy_v.append(inp_v_chunk[1][0])

        #preds_t = np.concatenate([test_batch(self.model, inp_t_i[0]) for inp_t_i in inp_t_chunks], axis=0)
        #preds_v = np.concatenate([test_batch(self.model, inp_v_i[0]) for inp_v_i in inp_v_chunks], axis=0)
        #print(type(inp_t_chunks))
        #print(len(inp_t_chunks))
        #print(len(inp_t_chunks[0][0]['RA']))
        #exit()
        #print(inp_t_chunks[0][1])
        
        #pred_t = np.sum(preds_t)
        #pred_v = np.sum(preds_v)

        mae_t = np.average(np.abs(np.array(preds_t) - np.array(energy_t)), axis=0)
        mae_v = np.average(np.abs(np.array(preds_v) - np.array(energy_v)), axis=0)
        
        loss_v_best = mae_v

        print(f"  (Pre-training)             MAE: {mae_t:>7.3f}/{mae_v:<7.3f}", flush=True)

        if model_path is not None:
            self.model.save(model_path)

        if False:
            learning_rate_scheduler = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=(num_batches * 60), decay_rate=0.5, staircase=True)
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate_scheduler)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = tf.keras.losses.MSE

        for ep in range(n_epochs):

            t1 = time.time()

            preds_t  = []
            err_t = []

            for batch in range(num_batches):
                batch_start = batch_size * batch
                inds_batch = inds_t[batch_start:min(Nt,batch_start+batch_size)]

                inp_batch, ie_batch = data_loader_t.get_data(inds_batch)

                preds_batch = train_batch(self.model, optimizer, loss_fn, inp_batch, ie_batch)
                #preds_batch = tf.reshape(preds_batch, [-1, 4])

                preds_t.append(preds_batch)
                #print(preds_batch)
                #print(ie_batch)
                err_t.append(preds_batch - ie_batch)


            #preds_t = np.concatenate(preds_t)
            #err_t = np.concatenate(err_t)
            mae_t = np.average(np.average(np.abs(err_t), axis=0))
            #total_mae_t = np.average(np.abs(np.sum(err_t, axis=1)))

            preds_v = []
            for inds in inds_v:
                inp_v_chunk = data_loader_v.get_data([inds])
                preds_v.append(np.sum(test_batch(self.model, inp_v_chunk[0])))
            #preds_v = np.array([np.sum(test_batch(self.model, inp_v_i[0])) for inp_v_i in inp_v_chunks])
            mae_v = np.average(np.average(np.abs(np.array(preds_v) - energy_v), axis=0))
            #total_mae_v = np.average(np.abs(np.sum(np.array(preds_v) - energy_v, axis=1)))

            loss_v = mae_v
            #total_loss_v = total_mae_v

            np.random.shuffle(inds_t)

            dt = time.time() - t1

            #if np.sum(loss_v) < np.sum(loss_v_best):
            if loss_v < loss_v_best:
                if model_path is not None:
                    self.model.save(model_path)
                loss_v_best = loss_v
                #total_loss_v_best = total_loss_v
                improved = "*"
            else:
                improved = ""

            #print(mae_t)
            #print(mae_v)
            print(f'EPOCH: {ep:4d} ({dt:<6.1f} sec)     MAE: {mae_t:>7.3f}/{mae_v:<7.3f} {improved}', flush=True)

        if log_path is not None:
            sys.stdout = default_stdout
            log_file.close()

    def predict(self, dimers):

        N = len(dimers)

        inds = np.arange(N)
        # TODO: replaced hardcoded 200 molecules. Probably want a data_loader.get_large_batch

        inds_chunks = [inds[i*200:min((i+1)*200,N)] for i in range(math.ceil(N / 200))]

        print("Processing Dataset...", flush=True)
        time_loaddata_start = time.time()
        data_loader = PairDataLoader(dimers, None, 5.0, self.model.get_config()["r_cut_im"], self.multipose)
        dt_loaddata = time.time() - time_loaddata_start
        print(f"...Done in {dt_loaddata:.2f} seconds", flush=True)

        print("\nPredicting Interaction Energies...", flush=True)
        time_predenergy_start = time.time()
        inp_chunks = [data_loader.get_data(inds_i) for inds_i in inds_chunks]
        preds = np.concatenate([test_batch(self.model, inp_i) for inp_i in inp_chunks], axis=0)
        dt_predenergy = time.time() - time_predenergy_start
        print(f"Done in {dt_predenergy:.2f} seconds", flush=True)

        return np.array(preds)

    # Possible TODO: predict_elst, transfer_learning, gradient




def softmax(values):
    values = np.log(1 + np.exp(values))
    value_weights = np.exp(values) / np.sum(np.exp(values))
    values *= value_weights
    return values


@tf.function(experimental_relax_shapes=True)
def train_batch(model, optimizer, loss_fn, inp, ie):

    #ie = tf.cast(ie, tf.float32)

    with tf.GradientTape() as tape:

        preds = model(inp, training=True)
        preds = tf.reduce_sum(preds)

        loss_value = loss_fn(ie, preds)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return preds

@tf.function(experimental_relax_shapes=True)
def test_batch(model, inp):

    preds  = model(inp, training=False)
    #preds = tf.reshape(preds, [-1, 4])

    return preds



if __name__ == "__main__":
    pass 
    #model = KerasPairModel()
    #model2 = PairModel()
