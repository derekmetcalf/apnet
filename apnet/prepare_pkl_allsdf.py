"""
This script is used to prepare pickles for AP-Net-dG training and inference.
Specifically, this will convert an sdf containing the protein as its first element and docked ligands
as all successive elements.
Requires ligands have an sdf property "training label" corresponding to the target quantity
"""

import os
import sys
import argparse
import pybel
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate dimers.pkl file for set of ligands stored as sdf with the protein as the first item')
    
    # Mandatory arguments
    parser.add_argument('system_sdf', help='[string] Path to sdf file')
    #parser.add_argument('label_pkl', help='[string] Path to corresponding pdb file')

    args = parser.parse_args(sys.argv[1:])
    sdf_path = args.system_sdf
    
    # Label path, in case we're looking at several ligand poses which share a label
    #label_path = args.label_pkl
    #label_df = pd.read_pickle(label_path)
    
    # Output to the same location as the pdb
    output_path = os.path.dirname(sdf_path)

    ligs = pybel.readfile('sdf', sdf_path)



    RA = []
    RB = []
    ZA = []
    ZB = []
    labels = []
    lab_nM = []
    systems = []
    pK_labels = False

    rb = []
    zb = []

    lig_dict = {}
    for i, lig in enumerate(ligs):
        if i == 0:
            for atom in lig.atoms:
                rb.append(list(atom.coords))
                zb.append(atom.atomicnum)
        else:    
            lig_name = lig.title
            lig_label = False
            if lig_name not in lig_dict:
                if 'training label' in lig.data.keys():
                    if len(lig.data['training label']) > 0:
                        labels.append(float(lig.data['training label']))
                        pK_labels = True
                        lig_label = True
                if lig_label:
                    systems.append(lig_name)
                    ra = []
                    za = []
                    for atom in lig.atoms:
                        ra.append(list(atom.coords))
                        za.append(atom.atomicnum)
                    RA.append(np.array(ra))
                    ZA.append(np.array(za))
                    ZB.append(np.array(zb))
                    RB.append(np.array(rb))

    if pK_labels:
        labels = np.array(labels)
    else:
        lab_nM = np.array(lab_nM)
        lab_M = lab_nM * 1e-9
        labels = -np.log10(lab_M)

    pair_data = {'RA':RA, 'ZA':ZA, 'RB':RB, 'ZB':ZB, 'label':labels, 'system':systems}
    df = pd.DataFrame(data=pair_data)
    df.to_pickle(os.path.join(output_path, 'dimers.pkl'))
    print(f'Successfully generated {os.path.join(output_path, "dimers.pkl")} for this dataset')
