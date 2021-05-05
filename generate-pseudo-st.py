import numpy as np
import pandas as pd
import pickle

# Load in single-cell matrix, labels
sc500 = np.load('sc_500.npy')
cellLabels = list(pd.read_csv('sc_cell_types.txt')['broad_type'])

nSpots = 500
nGenes = 500
nCells = sc500.shape[1]
pseudoSpotDict = {}

pseudoSpots = np.zeros((nGenes, nSpots))
for i in range(nSpots):
    nSpotCells = np.random.randint(2, 9)
    cells = np.random.randint(nCells, size=nSpotCells)
    pseudoSpots[:, i] = sc500[:, cells].sum(axis=1)
    pseudoSpotDict[i] = np.array(cellLabels)[cells]

np.save('pseudo_st_500.npy', pseudoSpots)
with open('pseudo_cell_types.pkl', 'wb') as f:
    pickle.dump(pseudoSpotDict, f)

