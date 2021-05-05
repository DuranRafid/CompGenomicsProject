# This is a sample Python script.

import numpy as np
import torch
from torch.utils.data import DataLoader
from train import train_network
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    STMat = np.load('st_500.npy')
    STMat = normalize(STMat, axis=0, norm='l1')
    SCMat = np.load('sc_500.npy')
    SCMat = normalize(SCMat, axis=0, norm='l1')
    number_of_genes,number_of_spots = STMat.shape
    number_of_genes1, number_of_cells = SCMat.shape

    assert number_of_genes == number_of_genes1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ST_Tensor = torch.from_numpy(STMat).float()
    SC_Tensor = torch.from_numpy(SCMat).float()
    ST_Tensor = torch.flatten(ST_Tensor).repeat(100,1)
    SC_Tensor = torch.flatten(SC_Tensor).repeat(100,1)

    SC_Dataloader = DataLoader(SC_Tensor, batch_size=2,shuffle=False)
    ST_Dataloader = DataLoader(ST_Tensor, batch_size=2,shuffle=False)
    dataloader =  zip(ST_Dataloader,SC_Dataloader)
    train_network(data_loader=dataloader,n_genes=number_of_genes,n_spots=number_of_spots,n_cells=number_of_cells,n_topic=20)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
