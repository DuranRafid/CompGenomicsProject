import numpy as np
import pandas as pd
import h5py

# Read in Mouse brain scRNAseq cell types
scRNAclusters = pd.read_csv('datasets/GSE71585_Clustering_Results.csv')
print(scRNAclusters.columns)
scRNAclusters.to_csv('sc_cell_types.txt', columns=['broad_type'])

# Read in TPM-normalized Mouse brain scRNAseq dataset
scRNApd = pd.read_csv('datasets/GSE71585_RefSeq_TPM.csv', index_col=0)
scRNAnp = scRNApd.to_numpy()

# Remove genes with 0 expression across all cells
indices = scRNAnp.mean(axis=1) > np.zeros(scRNAnp.shape[0])
scRNAnonzero = scRNAnp[indices, :]
scNames = list(scRNApd.index[indices])

# Read in h5 ST file
f = h5py.File('datasets/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5', 'r')

# Filter for genes that appear in at least 10% of all spots
# nGenes = f['matrix']['features']['id'].shape[0]
# nSpots = f['matrix']['barcodes'].shape[0]
# thr = np.int(.1 * nSpots)
# allGenes = np.array(f['matrix']['indices'])
# occuringGenes = []
# for gene in range(nGenes):
#     if gene % 100 == 0:
#         print(gene)
#     if np.count_nonzero(allGenes == gene) > thr:
#         occuringGenes.append(gene)
#
# # Save occuringGenes
# np.savetxt('present-in-10.txt', np.array(occuringGenes))

# Read in genes that are included in at least 10% of all spots
theTenPercenters = np.loadtxt('present-in-10.txt')
stBytes = list(f['matrix/features/name'][theTenPercenters])
stNames = [g.decode('utf-8') for g in stBytes]

# Create dictionary relating names to indices (will be useful later)
fdict = {}
for i in range(len(stNames)):
    fdict[stNames[i]] = theTenPercenters[i]

# Get genes present in both
intGenes = list(set(scNames) & set(stNames))

# Further select top 2000 highly expressed genes in scRNAseq dataset
scRNAintGenes = scRNApd.loc[intGenes]
scRNAigNP = scRNAintGenes.to_numpy()
cv = scRNAigNP.var(axis=1) / scRNAigNP.mean(axis=1)
cv_sort = np.argsort(cv)
top = cv_sort[-500:]
print(scRNAigNP.shape)
scRNAtop500 = scRNAigNP[top, :]
print(scRNAtop500.shape)

scRNAc = pd.read_csv('datasets/GSE71585_RefSeq_counts.csv', index_col=0)
scRNAcIntGenes = scRNAc.loc[intGenes]
print(scRNAcIntGenes.shape)
scRNActop = scRNAcIntGenes.to_numpy()[top, :]
print(scRNActop.shape)
np.save('datasets/sc_500.npy', scRNActop)

# Choose these genes in ST data, construct gene x spot matrix
genes = list(scRNAcIntGenes.index[top])
geneIdx = [np.int(fdict[g]) for g in genes]

# For each gene, iterate over spots and grab count
# nSpots = f['matrix/barcodes'].shape[0]
# stData = np.zeros((2000, nSpots))
#
# for j in range(nSpots):
#     if j % 100 == 0:
#         print(j)
#     leftIdx = f['matrix/indptr'][j]
#     rightIdx = f['matrix/indptr'][j+1]
#     spotGenes = list(f['matrix/indices'][leftIdx:rightIdx])
#     for i in range(2000):
#         gene = geneIdx[i]
#         if gene in spotGenes:
#             idx = spotGenes.index(gene)
#             spotCounts = list(f['matrix/data'][leftIdx:rightIdx])
#             stData[i, j] = spotCounts[idx]
#         else:
#             stData[i, j] = 0
#
# np.save('datasets/st_2000.npy', stData)

# Normalize stData with TPM normalization
print(genes)

stRaw = np.load('st_500.npy')

