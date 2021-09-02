""" Subclasses of the tensorflow.keras.Model class.
    These objects should be hidden from the user
 """

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

from apnet.layers import DistanceLayer, FeedForwardLayer, EdgeAttention

#################

max_Z = 100 # largest atomic number

#################

def get_distances(RA, RB, e_source, e_target):

    RA_source = tf.gather(RA, e_source)#, axis=1)
    RB_target = tf.gather(RB, e_target)#, axis=1)

    dR_xyz = RB_target - RA_source
    #dR_xyz = tf.squeeze(dR_xyz, axis=1)

    dR = tf.sqrt(tf.nn.relu(tf.reduce_sum(dR_xyz ** 2, -1)))

    return dR, dR_xyz

def get_messages(h0, h, rbf, e_source, e_target):

    nedge = tf.shape(e_source)[0]

    h0_source = tf.gather(h0, e_source)
    h0_target = tf.gather(h0, e_target)
    h_source = tf.gather(h, e_source)
    h_target = tf.gather(h, e_target)

    h_all = tf.concat([h0_source, h0_target, h_source, h_target], axis=-1)

    h_all_dot = tf.einsum('ez,er->ezr', h_all, rbf)
    h_all_dot = tf.reshape(h_all_dot, (nedge, -1))

    return tf.concat([h_all, h_all_dot, rbf], axis=-1)

def get_pair(hA, hB, rbf, e_source, e_target):

    hA_source = tf.gather(hA, e_source)
    hB_target = tf.gather(hB, e_target)

    # todo: outer product
    return tf.concat([hA_source, hB_target, rbf], axis=-1)

class KerasPairModel(tf.keras.Model):

    def __init__(self, atom_model=None, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut_im=8.0, scale_init=-1.e-2, shift_init=7.808, mode='lig-pair'):
        super(KerasPairModel, self).__init__()

        # pre-trained atomic model for predicting atomic properties
        self.atom_model = atom_model
        if self.atom_model is not None:
            self.atom_model.trainable = False

        # network hyperparameters
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.mode = mode
        #self.r_cut = 5.0

        # embed interatomic distances into large orthogonal basis
        self.distance_layer_im = DistanceLayer(n_rbf, r_cut_im)

        # embed atom types
        self.embed_layer = tf.keras.layers.Embedding(max_Z+1, n_embed)

        self.scale = tf.Variable(scale_init)
        self.shift = tf.Variable(shift_init)

        ## pre-trained atomic model for predicting atomic properties
        #self.atom_model = keras.models.load_model("/storage/home/hhive1/zglick3/data/test_apnet/atom_models/atom0/")
        #self.atom_model.trainable = False

        # the architecture contains many feed-forward dense nets with a tapered architecture
        layer_nodes_hidden = [n_neuron * 2, n_neuron, n_neuron // 2, n_embed]
        layer_nodes_readout = [n_neuron * 2, n_neuron, n_neuron // 2, 1]
        layer_activations = ["relu", "relu", "relu", "linear"]

        #self.readout_layer = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer')
        dropout = 0.2
        self.pair_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'pair_readout', dropout=dropout)
        self.lig_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'lig_readout', dropout=dropout)
        self.prot_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'prot_readout', dropout=dropout)

        # if desired, do a simple edge attention layer over the edges
        self.edge_dim = 2* (n_embed * 5) + n_rbf
        self.edge_attention = EdgeAttention(15, self.edge_dim, 'edge_attention', self.scale) 

        # embed distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, 5.0)

        self.update_layers = []
        self.readout_layers = []
        self.directional_layers = []
        self.directional_readout_layers = []

        for i in range(self.n_message):

            self.update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'update_layer_{i}'))
            self.readout_layers.append(FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_{i}'))

            self.directional_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'directional_layer_{i}'))
            self.directional_readout_layers.append(tf.keras.layers.Dense(1, activation='linear'))

    def mtp_elst(self, qA, muA, quadA, qB, muB, quadB, e_ABsr_source, e_ABsr_target, dR_ang, dR_xyz_ang):

        dR = dR_ang / 0.529177
        dR_xyz = dR_xyz_ang / 0.529177
        oodR = tf.math.reciprocal(dR)

        delta = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        qA_source = tf.gather(tf.squeeze(qA), e_ABsr_source)
        qB_source = tf.gather(tf.squeeze(qB), e_ABsr_target)

        muA_source = tf.gather(muA, e_ABsr_source)
        muB_source = tf.gather(muB, e_ABsr_target)

        quadA_source = (3.0 / 2.0) * tf.gather(quadA, e_ABsr_source)
        quadB_source = (3.0 / 2.0) * tf.gather(quadB, e_ABsr_target)

        E_qq = tf.einsum("x,x,x->x", qA_source, qB_source, oodR)
        
        T1 = tf.einsum('x,xy->xy', oodR ** 3, -1.0 * dR_xyz)
        qu = tf.einsum('x,xy->xy', qA_source, muB_source) - tf.einsum('x,xy->xy', qB_source, muA_source)
        E_qu = tf.einsum('xy,xy->x', T1, qu)

        T2 = 3 * tf.einsum('xy,xz->xyz', dR_xyz, dR_xyz) - tf.einsum('x,x,yz->xyz', dR, dR, delta)
        T2 = tf.einsum('x,xyz->xyz', oodR ** 5, T2)

        # this is basically zero?
        E_uu = -1.0 * tf.einsum('xy,xz,xyz->x', muA_source, muB_source, T2)

        qA_quadB_source = tf.einsum('x,xyz->xyz', qA_source, quadB_source)
        qB_quadA_source = tf.einsum('x,xyz->xyz', qB_source, quadA_source)
        E_qQ = tf.einsum('xyz,xyz->x', T2, qA_quadB_source + qB_quadA_source) / 3.0

        E_elst =  627.509 * (E_qq + E_qu + E_qQ + E_uu)
        #E_elst =  627.509 * (E_qq + E_qu)
        #E_elst =  627.509 * (E_qq + E_qu + E_uu)

        return E_elst




    def call(self, inputs):

        ########################
        ### unpack the input ###
        ########################

        # monomer atom coordinates and types
        ZA = inputs['ZA'][0]
        RA = inputs['RA'][0]
        ZB = inputs['ZB'][0]
        RB = inputs['RB'][0]

        # short range, intermolecular edges
        e_ABsr_source = inputs['e_ABsr_source'][0]
        e_ABsr_target = inputs['e_ABsr_target'][0]
        dimer_ind = inputs['dimer_ind'][0]

        # long range, intermolecular edges
        #e_ABlr_source = inputs['e_ABlr_source']
        #e_ABlr_target = inputs['e_ABlr_target']
        #dimer_ind_lr = inputs['dimer_ind_lr']

        # intramonomer edges (monomer A)
        e_AA_source = inputs['e_AA_source'][0]
        e_AA_target = inputs['e_AA_target'][0]

        # intramonomer edges (monomer B)
        e_BB_source = inputs['e_BB_source'][0]
        e_BB_target = inputs['e_BB_target'][0]

        # counts
        natomA = tf.shape(ZA)[0]
        natomB = tf.shape(ZB)[0]
        ndimer = tf.shape(inputs['total_charge_A'])[0]
        nedge_sr = tf.shape(e_ABsr_source)[0]
        #nedge_lr = tf.shape(e_ABlr_source)[0]

        # interatomic distances
        dR_sr, dR_sr_xyz = get_distances(RA, RB, e_ABsr_source, e_ABsr_target)
        #dR_lr, dR_lr_xyz = get_distances(RA, RB, e_ABlr_source, e_ABlr_target)
        dRA, dRA_xyz  = get_distances(RA, RA, e_AA_source, e_AA_target)
        dRB, dRB_xyz  = get_distances(RB, RB, e_BB_source, e_BB_target)

        # interatomic unit vectors
        dR_sr_unit = dR_sr_xyz / tf.expand_dims(dR_sr, -1)#tf.tile(tf.expand_dims(dR_sr, -1), [1, 1, 3])
        dRA_unit = dRA_xyz / tf.expand_dims(dRA, -1)
        dRB_unit = dRB_xyz / tf.expand_dims(dRB, -1)

        # distance encodings
        #rbf_sr = self.rbf_layer_im(dR_sr)
        rbf_sr = self.distance_layer_im(dR_sr)
        rbfA = self.distance_layer(dRA)
        rbfB = self.distance_layer(dRB)

        ##########################################################
        ### predict monomer properties w/ pretrained AtomModel ###
        ##########################################################

        inputsA = {
                'Z' : inputs['ZA'][0], #[0] hardcodes batch size 1
                'R' : inputs['RA'][0],
                'e_source' : inputs['e_AA_source'][0],
                'e_target' : inputs['e_AA_target'][0],
                'molecule_ind' : inputs['monomerA_ind'][0],
                'total_charge' : inputs['total_charge_A'][0]
        }

        inputsB = {
                'Z' : inputs['ZB'][0],
                'R' : inputs['RB'][0],
                'e_source' : inputs['e_BB_source'][0],
                'e_target' : inputs['e_BB_target'][0],
                'molecule_ind' : inputs['monomerB_ind'][0],
                'total_charge' : inputs['total_charge_B'][0]
        }

        qA, muA, quadA, hlistA = self.atom_model(inputsA)
        qB, muB, quadB, hlistB = self.atom_model(inputsB)
        
        atom_hA = tf.concat(hlistA, axis=-1)
        atom_hB = tf.concat(hlistB, axis=-1)


        ################################################################
        ### predict SAPT components via intramonomer message passing ###
        ################################################################
        
        # invariant hidden state lists
        # each list element is [natomA/B x nembed]
        hA_list = [tf.keras.layers.Flatten()(self.embed_layer(ZA))]
        hB_list = [tf.keras.layers.Flatten()(self.embed_layer(ZB))]
        #hA_list = tf.keras.layers.Flatten()(self.embed_layer(ZA))
        #hB_list = tf.keras.layers.Flatten()(self.embed_layer(ZB))
        hA_list.append(atom_hA)
        hB_list.append(atom_hB)


        ## directional hidden state lists
        ## each list element is [natomA/B x 3 x nembed]
        #hA_dir_list = []
        #hB_dir_list = []

        #for i in range(self.n_message):

        #    # intramonomer messages (from atom a to a' and from b to b')
        #    # [intrmonomer_edges x message_size]
        #    mA_ij = get_messages(hA_list[0], hA_list[-1], rbfA, e_AA_source, e_AA_target)
        #    mB_ij = get_messages(hB_list[0], hB_list[-1], rbfB, e_BB_source, e_BB_target)

        #    #################
        #    ### invariant ###
        #    #################

        #    # sum each atom's messages
        #    # [atoms x message_size]
        #    mA_i = tf.math.unsorted_segment_sum(mA_ij, e_AA_source, natomA)
        #    mB_i = tf.math.unsorted_segment_sum(mB_ij, e_BB_source, natomB)

        #    # get the next hidden state of the atom
        #    # [atomx x hidden_dim]
        #    hA_next = self.update_layers[i](mA_i)
        #    hB_next = self.update_layers[i](mB_i)

        #    hA_list.append(hA_next)
        #    hB_list.append(hB_next)

        #    ###################
        #    ### directional ###
        #    ###################

        #    # intromonomer directional messages are regular intramonomer messages, fed through a dense net
        #    mA_ij_dir = self.directional_layers[i](mA_ij) # [e x 8]
        #    mB_ij_dir = self.directional_layers[i](mB_ij) # [e x 8]

        #    # contract with intramonomer unit vectors to make directional
        #    mA_ij_dir = tf.einsum('ex,em->exm', dRA_unit, mA_ij_dir) # [e x 3 x 8]
        #    mB_ij_dir = tf.einsum('ex,em->exm', dRB_unit, mB_ij_dir) # [e x 3 x 8]

        #    # sum directional messages to get directional atomic hidden states
        #    # NOTE: this summation must be linear to guarantee equivariance.
        #    #       because of this constraint, we applied a dense net before the summation, not after
        #    hA_dir = tf.math.unsorted_segment_sum(mA_ij_dir, e_AA_source, natomA) # [a x 3 x 8]
        #    hB_dir = tf.math.unsorted_segment_sum(mB_ij_dir, e_BB_source, natomB) # [a x 3 x 8]

        #    hA_dir_list.append(hA_dir)
        #    hB_dir_list.append(hB_dir)

        ## concatenate hidden states over MP iterations
        hA = tf.keras.layers.Flatten()(tf.concat(hA_list, axis=-1))
        hB = tf.keras.layers.Flatten()(tf.concat(hB_list, axis=-1))

        # atom-pair features are a combo of atomic hidden states and the interatomic distance
        hAB = get_pair(hA, hB, rbf_sr, e_ABsr_source, e_ABsr_target)
        #hBA = get_pair(hB, hA, rbf_sr, e_ABsr_target, e_ABsr_source)


        ###########################
        ### binding free energy ###
        ###########################

        # run atom-pair features through a dense net to predict dG contributions
        # here, we're doing edge attention to get a system-wide feature vector that is read-out.
        # since we have a canonical protein-ligand choice for A and B, we can do a single pass
        
        sys_feats = hAB
        #sys_feats = self.edge_attention(hAB)
        #sys_feats = tf.expand_dims(sys_feats, axis=0)
        #pair_pred = self.pair_readout(pair_pred)
        lig_pred = self.lig_readout(hA)
        lig_pred = tf.reduce_sum(lig_pred, axis=0) * 0.0001
        
        if self.mode != "lig":
            pair_pred = self.pair_readout(sys_feats)
            pair_pred = tf.reduce_sum(pair_pred, axis=0) * 3e-6
            
            if self.mode == "prot-lig-pair":
                prot_pred = self.prot_readout(hB)
                prot_pred = tf.reduce_sum(prot_pred, axis=0) * 0.0001
 
                dG_pred = pair_pred - prot_pred - lig_pred + self.shift
                return dG_pred
            else:
                dG_pred = pair_pred - lig_pred + self.shift
                return dG_pred
        else:
            dG_pred = lig_pred + self.shift
            return dG_pred


    def get_config(self):

        return {
            #"atom_model" : self.atom_model,
            "n_message" : self.n_message,
            "n_rbf" : self.n_rbf,
            "n_neuron" : self.n_neuron,
            "n_embed" : self.n_embed,
            "r_cut_im" : self.r_cut_im,
        }

    @classmethod
    def from_config(cls, config):
        return cls(#atom_model=config["atom_model"],
                   n_message=config["n_message"],
                   n_rbf=config["n_rbf"],
                   n_neuron=config["n_neuron"],
                   n_embed=config["n_embed"],
                   r_cut_im=config["r_cut_im"])


class KerasDeltaModel(tf.keras.Model):

    def __init__(self, pair_model, atom_model, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut_im=8.0, scale_init=-1.e-2, shift_init=7.808, mode='lig-pair'):
        super(KerasDeltaModel, self).__init__()

        # pre-trained atomic model for predicting atomic properties
        self.atom_model = atom_model
        self.atom_model.trainable = False
        self.pair_model = pair_model
        self.pair_model.trainable = False

        # network hyperparameters
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.mode = mode
        #self.r_cut = 5.0

        # embed interatomic distances into large orthogonal basis
        self.distance_layer_im = DistanceLayer(n_rbf, r_cut_im)

        # embed atom types
        self.embed_layer = tf.keras.layers.Embedding(max_Z+1, n_embed)

        self.scale = tf.Variable(scale_init)

        ## pre-trained atomic model for predicting atomic properties
        #self.atom_model = keras.models.load_model("/storage/home/hhive1/zglick3/data/test_apnet/atom_models/atom0/")
        #self.atom_model.trainable = False

        # the architecture contains many feed-forward dense nets with a tapered architecture
        layer_nodes_hidden = [n_neuron * 2, n_neuron, n_neuron // 2, n_embed]
        layer_nodes_readout = [n_neuron * 2, n_neuron, n_neuron // 2, 1]
        layer_activations = ["relu", "relu", "relu", "linear"]

        #self.readout_layer = FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer')
        dropout = 0.2
        self.pair_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'pair_readout', dropout=dropout)
        self.lig_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'lig_readout', dropout=dropout)
        self.prot_readout = FeedForwardLayer(layer_nodes_readout, layer_activations, f'prot_readout', dropout=dropout)

        # if desired, do a simple edge attention layer over the edges
        self.edge_dim = 2* (n_embed * 5) + n_rbf
        self.edge_attention = EdgeAttention(15, self.edge_dim, 'edge_attention', self.scale) 

        # embed distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, 5.0)

        self.update_layers = []
        self.readout_layers = []
        self.directional_layers = []
        self.directional_readout_layers = []

        for i in range(self.n_message):

            self.update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'update_layer_{i}'))
            self.readout_layers.append(FeedForwardLayer(layer_nodes_readout, layer_activations, f'readout_layer_{i}'))

            self.directional_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations, f'directional_layer_{i}'))
            self.directional_readout_layers.append(tf.keras.layers.Dense(1, activation='linear'))

    def call(self, inputs):

        ########################
        ### unpack the input ###
        ########################

        # monomer atom coordinates and types
        ZA = inputs['ZA'][0]
        RA = inputs['RA'][0]
        ZB = inputs['ZB'][0]
        RB = inputs['RB'][0]

        # short range, intermolecular edges
        e_ABsr_source = inputs['e_ABsr_source'][0]
        e_ABsr_target = inputs['e_ABsr_target'][0]
        dimer_ind = inputs['dimer_ind'][0]

        # long range, intermolecular edges
        #e_ABlr_source = inputs['e_ABlr_source']
        #e_ABlr_target = inputs['e_ABlr_target']
        #dimer_ind_lr = inputs['dimer_ind_lr']

        # intramonomer edges (monomer A)
        e_AA_source = inputs['e_AA_source'][0]
        e_AA_target = inputs['e_AA_target'][0]

        # intramonomer edges (monomer B)
        e_BB_source = inputs['e_BB_source'][0]
        e_BB_target = inputs['e_BB_target'][0]

        # counts
        natomA = tf.shape(ZA)[0]
        natomB = tf.shape(ZB)[0]
        ndimer = tf.shape(inputs['total_charge_A'])[0]
        nedge_sr = tf.shape(e_ABsr_source)[0]
        #nedge_lr = tf.shape(e_ABlr_source)[0]

        # interatomic distances
        dR_sr, dR_sr_xyz = get_distances(RA, RB, e_ABsr_source, e_ABsr_target)
        #dR_lr, dR_lr_xyz = get_distances(RA, RB, e_ABlr_source, e_ABlr_target)
        dRA, dRA_xyz  = get_distances(RA, RA, e_AA_source, e_AA_target)
        dRB, dRB_xyz  = get_distances(RB, RB, e_BB_source, e_BB_target)

        # interatomic unit vectors
        dR_sr_unit = dR_sr_xyz / tf.expand_dims(dR_sr, -1)#tf.tile(tf.expand_dims(dR_sr, -1), [1, 1, 3])
        dRA_unit = dRA_xyz / tf.expand_dims(dRA, -1)
        dRB_unit = dRB_xyz / tf.expand_dims(dRB, -1)

        # distance encodings
        #rbf_sr = self.rbf_layer_im(dR_sr)
        rbf_sr = self.distance_layer_im(dR_sr)
        rbfA = self.distance_layer(dRA)
        rbfB = self.distance_layer(dRB)

        ##########################################################
        ### predict monomer properties w/ pretrained AtomModel ###
        ##########################################################

        inputsA = {
                'Z' : inputs['ZA'][0], #[0] hardcodes batch size 1
                'R' : inputs['RA'][0],
                'e_source' : inputs['e_AA_source'][0],
                'e_target' : inputs['e_AA_target'][0],
                'molecule_ind' : inputs['monomerA_ind'][0],
                'total_charge' : inputs['total_charge_A'][0]
        }

        inputsB = {
                'Z' : inputs['ZB'][0],
                'R' : inputs['RB'][0],
                'e_source' : inputs['e_BB_source'][0],
                'e_target' : inputs['e_BB_target'][0],
                'molecule_ind' : inputs['monomerB_ind'][0],
                'total_charge' : inputs['total_charge_B'][0]
        }

        qA, muA, quadA, hlistA = self.atom_model(inputsA)
        qB, muB, quadB, hlistB = self.atom_model(inputsB)
        
        atom_hA = tf.concat(hlistA, axis=-1)
        atom_hB = tf.concat(hlistB, axis=-1)


        ################################################################
        ### predict SAPT components via intramonomer message passing ###
        ################################################################
        
        # invariant hidden state lists
        # each list element is [natomA/B x nembed]
        hA_list = [tf.keras.layers.Flatten()(self.embed_layer(ZA))]
        hB_list = [tf.keras.layers.Flatten()(self.embed_layer(ZB))]
        hA_list.append(atom_hA)
        hB_list.append(atom_hB)

        ## directional hidden state lists
        ## each list element is [natomA/B x 3 x nembed]
        #hA_dir_list = []
        #hB_dir_list = []

        #for i in range(self.n_message):

        #    # intramonomer messages (from atom a to a' and from b to b')
        #    # [intrmonomer_edges x message_size]
        #    mA_ij = get_messages(hA_list[0], hA_list[-1], rbfA, e_AA_source, e_AA_target)
        #    mB_ij = get_messages(hB_list[0], hB_list[-1], rbfB, e_BB_source, e_BB_target)

        #    #################
        #    ### invariant ###
        #    #################

        #    # sum each atom's messages
        #    # [atoms x message_size]
        #    mA_i = tf.math.unsorted_segment_sum(mA_ij, e_AA_source, natomA)
        #    mB_i = tf.math.unsorted_segment_sum(mB_ij, e_BB_source, natomB)

        #    # get the next hidden state of the atom
        #    # [atomx x hidden_dim]
        #    hA_next = self.update_layers[i](mA_i)
        #    hB_next = self.update_layers[i](mB_i)

        #    hA_list.append(hA_next)
        #    hB_list.append(hB_next)


        ## concatenate hidden states over MP iterations
        hA = tf.keras.layers.Flatten()(tf.concat(hA_list, axis=-1))
        hB = tf.keras.layers.Flatten()(tf.concat(hB_list, axis=-1))

        # atom-pair features are a combo of atomic hidden states and the interatomic distance
        hAB = get_pair(hA, hB, rbf_sr, e_ABsr_source, e_ABsr_target)
        #hBA = get_pair(hB, hA, rbf_sr, e_ABsr_target, e_ABsr_source)


        ###########################
        ### binding free energy ###
        ###########################

        # run atom-pair features through a dense net to predict dG contributions
        # since we have a canonical protein-ligand choice for A and B, we can do a single pass
        
        sys_feats = hAB
        #sys_feats = self.edge_attention(hAB)
        #sys_feats = tf.expand_dims(sys_feats, axis=0)
        #pair_pred = self.pair_readout(pair_pred)

        base_model_pred = self.pair_model(inputs) 
        base_model_pred = tf.reduce_sum(base_model_pred)

        pair_pred = self.pair_readout(sys_feats)
        pair_pred = tf.reduce_sum(pair_pred, axis=0) * 3e-6
        
        if self.mode == "prot-lig-pair":
            prot_pred = self.prot_readout(hB)
            prot_pred = tf.reduce_sum(prot_pred, axis=0) * 0.0001
 
            dG_pred = pair_pred - prot_pred + base_model_pred
            return dG_pred
        elif self.mode == "lig-pair":
            dG_pred = pair_pred + base_model_pred
            return dG_pred
        else:
            dG_pred = base_model_pred
            return dG_pred


    def get_config(self):

        return {
            #"atom_model" : self.atom_model,
            "n_message" : self.n_message,
            "n_rbf" : self.n_rbf,
            "n_neuron" : self.n_neuron,
            "n_embed" : self.n_embed,
            "r_cut_im" : self.r_cut_im,
        }

    @classmethod
    def from_config(cls, config):
        return cls(#atom_model=config["atom_model"],
                   n_message=config["n_message"],
                   n_rbf=config["n_rbf"],
                   n_neuron=config["n_neuron"],
                   n_embed=config["n_embed"],
                   r_cut_im=config["r_cut_im"])

if __name__ == "__main__":

    model = KerasPairModel()
