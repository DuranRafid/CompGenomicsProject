# Takes in the spot x topic and topic x cell matrices
# Multiplies the two matrices together to obtain the spot x cell matrix
# Performs various evaluations on the resulting matrix
import numpy as np
import math

class SpotCellEval:
    def __init__(self, spot_by_topic, topic_by_cell):
        self.spot_by_topic = spot_by_topic
        self.topic_by_cell = topic_by_cell
        self.spot_by_cell = self.getSpotByCell()

    def getSpotByCell(self):
        return np.dot(self.spot_by_topic, self.topic_by_cell)

    # Calculates Kl-divergence
    def KLD(self, p,q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    # Calculates JSD metric
    def JSD(self, p, q): 
        m = 0.5 * (p + q) 
        return 0.5 * self.KLD(p, m) + 0.5 * self.KLD(q, m)

    def JSD_dist(self, p, q):
        return np.sqrt(self.JSD(p,q))

    def test_eval(self, truth_mat):
        jsd_matrix = np.zeros(shape=(truth_mat.shape[0], 1), dtype=float)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(truth_mat.shape[0]):
            p = truth_mat[i,:]
            q = self.spot_by_cell[i,:]

            if  np.sum(truth_mat[i,:]) > 0:
                jsd_matrix[i,0] = self.JSD(p,q)
            else:
                jsd_matrix[i,0] = 1

            for j in range(truth_mat.shape[1]):
                if p[j] > 0 and q[j] > 0:
                    tp += 1
                elif p[j] == 0 and q[j]==0:
                    tn += 1
                elif p[j] > 0 and q[j] == 0:
                    fn += 1
                elif p[j] == 0 and q[j] > 0:
                    fp += 1

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1 = 2*((precision*recall)/(precision + recall))

        '''
        accuracy = 0
        sensitivity = 0
        specificity = 0
        precision = 0
        recall = 0
        F1 = 0
        '''

        quant = np.array([0.25, 0.5, 0.75])
        quantile_jsd = np.quantile(jsd_matrix, quant, axis=0)

        metrics = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, recall, F1, quantile_jsd]

        return jsd_matrix, metrics