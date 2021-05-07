import numpy as np
import pandas as pd
import h5py

class SCST():
    def __init__(self, scTpmPath, scCountPath, scMetaPath, stPath):
        # Read in datasets
        self.scTpmDf = pd.read_csv(scTpmPath, index_col=0)
        self.scCountDf = pd.read_csv(scCountPath, index_col=0)
        self.scMetaDf = pd.read_csv(scMetaPath, sep=',')
        self.stH5 = h5py.File(stPath, 'r')

        # Initialize important parameters
        self.nSpots = self.stH5['matrix/barcodes'].shape[0]
        self.nCells = self.scTpmDf.shape[1]
        cTypeList = list(self.scMetaDf['broad_type'])
        self.cellTypes, self.cellTypeIndices = np.unique(cTypeList, return_inverse=True)
        self.nCellTypes = len(self.cellTypes)

        # ST-specific variables
        self.stGenes = []
        self.stDict = {}

        # SC-specific variables
        self.genes = None
        self.nGenes = 0

    def filter_st_genes(self, pcnt, verbose=True):
        '''
        Filter out ST genes by % threshold
        Genes not occurring in at least % of spots filtered out
        Creates dictionary of filtered gene names and indices in self.sth5
        :param pcnt: Percentage of spots genes must be in
        :param verbose: Flag, if True, indicates to print progress in filtering
        '''
        # Get total st genes, gene indices over all spots
        nGenes = self.stH5['matrix/features/id'].shape[0]
        allGenes = np.array(self.stH5['matrix/indices'])

        # Set threshold for minimum number of spots
        thr = np.int(pcnt * self.nSpots / 100)

        # Count number of spots gene appears in, append if above thr
        if verbose:
            for gene in range(nGenes):
                if gene % 1000 == 0:
                    print('Starting gene ' + str(gene+1) + '/' + str(nGenes))
                if np.count_nonzero(allGenes == gene) > thr:
                    self.stGenes.append(gene)
        else:
            for gene in range(nGenes):
                if np.count_nonzero(allGenes == gene) > thr:
                    self.stGenes.append(gene)

        # Get gene names for frequently occurring genes
        stBytes = list(self.stH5['matrix/features/name'][self.stGenes])
        stNames = [g.decode('utf-8') for g in stBytes]

        # Create dictionary (key = gene name, value = index)
        for i in range(len(stNames)):
            self.stDict[stNames[i]] = self.stGenes[i]

    def filter_sc_genes(self, nGenes):
        '''
        Filter out sc genes by biological variance
        Use coefficient of variation
        To be run after filter_st_genes
        :param nGenes: Total number of genes to choose
        :return gene x cell matrix
        '''
        # Remove genes with 0 expression across all cells
        indices = np.array(self.scTpmDf).mean(axis=1) > np.zeros(self.scTpmDf.shape[0])
        scNonzeroNames = list(self.scTpmDf.index[indices])
        scTpmNonzero = self.scTpmDf.loc[scNonzeroNames]

        # Filter out genes already filtered out in filter_st_genes
        stNames = self.stDict.keys()
        scStGenes = list(set(stNames) & set(scNonzeroNames))

        # Further select top nGenes highly expressed genes in scRNAseq dataset
        scStTpm = scTpmNonzero.loc[scStGenes]
        cv = np.array(scStTpm).var(axis=1) / np.array(scStTpm).mean(axis=1)
        cvSort = np.argsort(cv)
        topGenes = cvSort[-nGenes:]

        # Save final filtered gene names, gene x cell single-cell matrix
        self.genes = scStTpm.index[topGenes]
        self.nGenes = len(self.genes)
        scMat = np.array(self.scCountDf.loc[self.genes])
        return(scMat)

    # SLOW VERSION, COMMENTED OUT IN CASE I NEED PARTS OF IT LATER
    # def construct_st_mat(self, verbose=True):
    #     '''
    #     Construct ST matrix from final gene list
    #     Should be run after filter_st_genes, filter_sc_genes
    #     :param verbose: Flag, default true
    #     :return gene x spot matrix
    #     '''
    #     # Initialize stMat, get gene indices of final filtered genes
    #     stMat = np.zeros((self.nGenes, self.nSpots))
    #     geneIdx = [np.int(self.stDict[g]) for g in self.genes]
    #
    #     # If wanting updates, use this for loop (minimize checking if statements)
    #     if verbose:
    #         for j in range(self.nSpots):
    #             if j % 100 == 0:
    #                 print('Starting spot ' + str(j+1) + '/' + str(self.nSpots))
    #             leftIdx = self.stH5['matrix/indptr'][j]
    #             rightIdx = self.stH5['matrix/indptr'][j+1]
    #             spotGenes = list(self.stH5['matrix/indices'][leftIdx:rightIdx])
    #
    #             # For each gene, check if gene in given spot
    #             # If so, record gene count
    #             for i in range(self.nGenes):
    #                 gene = geneIdx[i]
    #                 if gene in spotGenes:
    #                     idx = spotGenes.index(gene)
    #                     spotCounts = list(self.stH5['matrix/data'][leftIdx:rightIdx])
    #                     stMat[i, j] = spotCounts[idx]
    #     else:
    #         for j in range(self.nSpots):
    #             leftIdx = self.stH5['matrix/indptr'][j]
    #             rightIdx = self.stH5['matrix/indptr'][j+1]
    #             spotGenes = list(self.stH5['matrix/indices'][leftIdx:rightIdx])
    #
    #             # For each gene, check if gene in given spot
    #             # If so, record gene count
    #             for i in range(self.nGenes):
    #                 gene = geneIdx[i]
    #                 if gene in spotGenes:
    #                     idx = spotGenes.index(gene)
    #                     spotCounts = list(self.stH5['matrix/data'][leftIdx:rightIdx])
    #                     stMat[i, j] = spotCounts[idx]
    #
    #     return(stMat)

    def construct_st_mat(self, verbose=True):
        '''
        Construct ST matrix from final gene list
        Should be run after filter_st_genes, filter_sc_genes
        :param verbose: Flag, default true
        :return gene x spot matrix
        '''
        # Initialize stMat, get gene indices of final filtered genes
        stMat = np.zeros((self.nGenes, self.nSpots))
        geneIdx = [np.int(self.stDict[g]) for g in self.genes]

        # If wanting updates, use this for loop (minimize checking if statements)
        if verbose:
            for i in range(self.nGenes):
                if i % 100 == 0:
                    print('Starting gene ' + str(i + 1) + '/' + str(self.nGenes))
                # Find all instances of gene
                gene = geneIdx[i]
                idxs = np.where(np.array(self.stH5['matrix/indices']) == gene)[0]

                # Associate indices with spots
                ptrs = list(self.stH5['matrix/indptr'])
                last = 0
                spots = []
                for j in range(len(idxs)):
                    for k in range(last, len(ptrs)):
                        if idxs[j] < ptrs[k]:
                            spots.append(k-1)
                            last = k - 1
                            break

                # Fill in row of matrix
                stMat[i, spots] = self.stH5['matrix/data'][idxs]

        else:
            for i in range(self.nGenes):
                # Find all instances of gene
                gene = geneIdx[i]
                idxs = np.where(np.array(self.stH5['matrix/indices']) == gene)[0]

                # Associate indices with spots
                ptrs = list(self.stH5['matrix/indptr'])
                spots = []
                for idx in idxs:
                    for j in range(len(ptrs)):
                        if idx < ptrs[j]:
                            spots.append(j-1)
                            break

                # Fill in row of matrix
                stMat[i, spots] = self.stH5['matrix/data'][idxs]

        return(stMat)

    def generate_pseudo_st(self, scMat):
        '''
        Constructs pseudo-ST data from SC data matrix
        :return: pseudo-ST gene x spot matrix, ground truth cell type x spot matrix
        '''
        pseudoSpots = np.zeros((self.nGenes, self.nSpots))
        trueSpotComp = np.zeros((self.nCellTypes, self.nSpots))
        for i in range(self.nSpots):
            nSpotCells = np.random.randint(2, 9)
            cells = np.random.randint(self.nCells, size=nSpotCells)
            pseudoSpots[:, i] = scMat[:, cells].sum(axis=1)
            for cell in cells:
                trueSpotComp[self.cellTypeIndices[cell], i] += 1

        return(pseudoSpots, trueSpotComp)