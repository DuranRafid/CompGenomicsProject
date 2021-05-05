import numpy as np
if __name__=="__main__":
    SC_mat = np.load('sc_500.npy')
    print(SC_mat.shape)
    ST_mat = np.load('st_500.npy')
    print(ST_mat.shape)