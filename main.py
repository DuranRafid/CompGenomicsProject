# This is a sample Python script.

import numpy as np
import torch
from torch.utils.data import DataLoader
from train import train_network
from sklearn.preprocessing import normalize
from data import get_data_loaders
if __name__ == '__main__':
    number_of_genes = 2000
    number_of_spots = 3355
    number_of_cells = 1809
    train_loader, test_loader = get_data_loaders()
    train_network(train_data_loader=train_loader, test_data_loader= test_loader,n_genes=number_of_genes,n_spots=number_of_spots,n_cells=number_of_cells,n_topic=8)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
