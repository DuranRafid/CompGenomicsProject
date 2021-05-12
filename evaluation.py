# Takes in the spot x topic and topic x cell matrices
# Multiplies the two matrices together to obtain the spot x cell matrix
# Performs various evaluations on the resulting matrix
import torch
import numpy as np
import math
from scipy.optimize import nnls
from scipy.stats import entropy

class SpotCellEval:
    def __init__(self, spot_by_topic, topic_by_cell, labels):
        self.spot_by_topic = spot_by_topic
        self.topic_by_cell = topic_by_cell
        self.topic_by_cell_type = self.convertCellToType(labels)
        self.spot_by_cell_type = self.getSpotByCellType()

    def getSpotByCellType(self, train=True):  # If train is true, use NNLS to minimize residuals
        if train:
            Atens = self.topic_by_cell_type.detach().numpy()
            Btens = self.spot_by_topic.detach().numpy()
            spotByCellType = np.zeros((Atens.shape[0], Atens.shape[2], Btens.shape[2]))
            self.residual = 0
            for i in range(Atens.shape[0]):
                A = Atens[i, :, :].reshape(-1, Atens.shape[2])
                B = Btens[i, :, :].reshape(-1, Btens.shape[2]).T
                for spot in range(B.shape[1]):
                    spotByCellType[i, :, spot], residual = nnls(A, B[:, spot])
                    self.residual += residual
            return spotByCellType
        else:
            return np.dot(self.spot_by_topic.detach().numpy(), self.topic_by_cell_type.detach().numpy())

    def convertCellToType(self, labels):
        batch_size = np.int(self.topic_by_cell.size(0))
        num_topics = np.int(self.topic_by_cell.shape[2])
        mat = torch.zeros(batch_size, num_topics, 8)
        for i in range(batch_size):
            for j in range(num_topics):
                sum = 0
                for k in range(8):
                    new = torch.mean(self.topic_by_cell[i, labels == k, j])
                    mat[i, j, k] = new
                    sum += new
                mat[i, j, :] /= sum
        return mat

    # Calculates Kl-divergence
    def KLD(self, p,q):
        #print(np.log(p/q))
        #return np.sum(np.where(p != 0, p * np.log(p / q), 0))
        temp = p
        for i in range(p.shape[0]):
            if p[i]!=0:
                num = p[i] * np.log2(p[i]/q[i])
                if np.isnan(num):
                    temp[i] = 0
                else:
                    temp[i] = num
            else:
                temp[i] = 0
        return np.sum(temp)

    # Calculates JSD metric
    #def JSD(self, p, q): 
    #    m = 0.5 * (p + q) 
    #return 0.5 * self.KLD(p, m) + 0.5 * self.KLD(q, m)

    def JSD(self, p, q, base=np.e):
        '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        '''
        ## convert to np.array
        #p, q = np.asarray(p), np.asarray(q)
         ## normalize p, q to probabilities
        #p, q = p/p.sum(), q/q.sum()
        m = 1./2*(p + q)
        return entropy(p,m, base=base)/2. +  entropy(q, m, base=base)/2.

    def JSD_dist(self, p, q):
        return np.sqrt(self.JSD(p,q))

    def test_eval(self, truth_mat):
        jsd_matrix = np.zeros(shape=(truth_mat.shape[0], truth_mat.shape[1], 1), dtype=float)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for b in range(truth_mat.shape[0]):
            print(truth_mat.shape)
            p = truth_mat.detach().numpy()
            p = p[b,:,:].reshape(-1, p.shape[2])
            #q = self.spot_by_cell_type[i,:]
            q = self.spot_by_cell_type
            q = q[b,:,:].reshape(-1, q.shape[2])

            print(p.shape)
            print(q.shape)
            for i in range(p.shape[0]):
                #if  torch.sum(truth_mat[i,:]) > 0:
                if np.sum(p[i,:]) > 0:
                    #print(self.JSD(p[i,:],q[i,:]))
                    jsd_matrix[b,i,0] = self.JSD(p[i,:],q[i,:])
                else:
                    jsd_matrix[b,i,0] = 1

                for j in range(p.shape[1]):
                    if p[i,j] > 0 and q[i,j] > 0:
                        tp += 1
                    elif p[i,j] == 0 and q[i,j]==0:
                        tn += 1
                    elif p[i,j] > 0 and q[i,j] == 0:
                        fn += 1
                    elif p[i,j] == 0 and q[i,j] > 0:
                        fp += 1
        
        '''
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        sensitivity = tp/(tp+fn)
        #specificity = tn/(tn+fp)
        #precision = tp/(tp+fp)
        #recall = tp/(tp+fn)
        #F1 = 2*((precision*recall)/(precision + recall))

        
        #accuracy = 0
        #sensitivity = 0
        specificity = 0
        precision = 0
        recall = 0
        F1 = 0
        '''

        quant = np.array([0.25, 0.5, 0.75])
        quantile_jsd = np.quantile(jsd_matrix.mean(axis=0), quant, axis=0)

        #metrics = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, recall, F1, quantile_jsd]
        metrics = [tp, tn, fp, fn, quantile_jsd]

        return jsd_matrix.mean(axis=0), metrics
