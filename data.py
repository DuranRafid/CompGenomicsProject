import numpy as np
from sklearn.preprocessing import  normalize
if __name__=="__main__":
    SC_mat = np.load('sc_500.npy')
    SC_mat = normalize(SC_mat,axis=0, norm='l1')
    print(sum(SC_mat[:,0]))
    ST_mat = np.load('st_500.npy')
    ST_mat = normalize(ST_mat,axis=0, norm='l1')
    print(ST_mat[0])