import os
import sys
import argparse
import pybel
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate dimers.pkl file for set of ligands stored as sdf and protein as pdb')
    
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
            #print(lig.data)
            lig_name = lig.title
            lig_label = False
            if lig_name not in lig_dict:
                #print(lig_name)
                #lig_dict[lig_name] = 1
                #labels.append(label_df.loc[label_df['system'] == lig_name, 'label'])
                #print(lig.data.keys())
                #if 'Ki (nM)' in lig.data.keys():
                #    if len(lig.data['Ki (nM)']) > 0:
                #        lab_nM.append(float(lig.data['Ki (nM)'].strip('<').strip('>')))
                #        lig_label = True
                #elif 'IC50 (nM)' in lig.data.keys():
                #    if len(lig.data['IC50 (nM)']) > 0:
                #        lab_nM.append(float(lig.data['IC50 (nM)'].strip('<').strip('>')))
                #        lig_label = True
                if 'training label' in lig.data.keys():
                    if len(lig.data['training label']) > 0:
                        labels.append(float(lig.data['training label']))
                        pK_labels = True
                        lig_label = True
                        print(lig.data['training label'])
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

        #else:
        #    lig_dict[lig_name] = lig_dict[lig_name] + 1
    if pK_labels:
        labels = np.array(labels)
    else:
        lab_nM = np.array(lab_nM)
        lab_M = lab_nM * 1e-9
        labels = -np.log10(lab_M)
    #all_ras = []
    #for unique_ligand in systems:
    #    #print(unique_ligand)
    #    sys_ras = []
    #    #print(ligs[0].title)
    #    ligs = pybel.readfile('sdf', sdf_path)
    #    for lig in ligs:
    #        #lig_name = lig.title
    #        #print(lig_name)
    #        #print(unique_ligand, lig_name)
    #        #if unique_ligand == lig_name:
    #        #print(unique_ligand)
    #        ra = []
    #        for atom in lig.atoms:
    #            ra.append(atom.coords)
    #        sys_ras.append(np.array(ra))
    #    all_ras.append(np.array(sys_ras))
    #RA = all_ras

    RA = RA
    pair_data = {'RA':RA, 'ZA':ZA, 'RB':RB, 'ZB':ZB, 'label':labels, 'system':systems}
    print(len(RA), len(ZA), len(RB), len(ZB), len(labels), len(systems), len(lig_dict))
    df = pd.DataFrame(data=pair_data)
    df.to_pickle(os.path.join(output_path, 'dimers.pkl'))
    print(f'Successfully generated {os.path.join(output_path, "dimers.pkl")} for this dataset')
