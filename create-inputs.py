import numpy as np
import pandas as pd
import h5py
from SCSTProcessing import SCST

# Get input paths
scTpm = 'datasets/GSE71585_RefSeq_TPM.csv'
scCount = 'datasets/GSE71585_RefSeq_counts.csv'
scMeta = 'datasets/GSE71585_Clustering_Results.csv'
stPost1 = 'datasets/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5'
stPost2 = 'datasets/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix.h5'
stAnt1 = 'datasets/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5'
stAnt2 = 'datasets/V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix.h5'

stIns = [stPost1, stPost2, stAnt1, stAnt2]
desc = ['post-1', 'post-2', 'ant-1', 'ant-2']
scOuts = ['input-matrices/sc-2000-' + d + '.npy' for d in desc]
stOuts = ['input-matrices/st-2000-' + d + '.npy' for d in desc]
pstOuts = ['input-matrices/pst-2000-' + d + '-' for d in desc]

for i in range(4):
    print('Processing dataset pair ' + str(i+1))
    processing = SCST(scTpm, scCount, scMeta, stIns[i])
    print('Successfully read in data')
    print('Filtering genes in ST data')
    processing.filter_st_genes(10)
    print('Successfully filtered genes in ST')
    print('Filtering genes in SC data')
    scMat = processing.filter_sc_genes(2000)
    print('Successfully filtered genes in SC')
    print('Constructing ST matrix')
    stMat = processing.construct_st_mat()
    print('Successfully constructed ST matrix')
    print('Saving SC, ST matrices')
    np.save(scOuts[i], scMat)
    np.save(stOuts[i], stMat)
    print('Saved SC, ST matrices')
    print('Generating pseudo-ST matrices, ground truth')
    pstOutReplicates = [pstOuts[i] + str(k) + '.npy' for k in range(100)]
    pstOutTruths = [pstOuts[i] + str(k) + '-truth.npy' for k in range(100)]
    for j in range(100):
        pstMat, trueTypes = processing.generate_pseudo_st(scMat)
        np.save(pstOutReplicates[j], pstMat)
        np.save(pstOutTruths[j], trueTypes)
    print('Successfully generated pseudo-ST matrices, ground truth')