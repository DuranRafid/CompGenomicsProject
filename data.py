import numpy as np
from sklearn.preprocessing import  normalize
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def get_data_loaders():
    mat_dir = os.path.join(os.getcwd(),"input-matrices")
    input_file_list = [file for file in os.listdir(mat_dir) if 'pst' in file]
    label_mat_list = [file for file in input_file_list if 'truth' in file]
    data_mat_list = [file for file in input_file_list if 'truth' not in file]
    input_data = []
    for filename in data_mat_list:
        filepath = os.path.join(mat_dir,filename)
        input_data.append(normalize(np.load(filepath), axis=0, norm='l1'))
    input_data = np.array(input_data)
    input_data = input_data.reshape(100, -1)

    input_label = []
    for filename in label_mat_list:
        filepath = os.path.join(mat_dir, filename)
        input_label.append(np.load(filepath).T)
    input_label = np.array(input_label)

    X_train, X_test, y_train, y_test = train_test_split(input_data, input_label, test_size=0.2, random_state=0)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    SC_mat = np.load('datasets/sc_2000.npy')
    SC_mat = normalize(SC_mat, axis=0, norm='l1')

    X_ref = torch.from_numpy(SC_mat).float()
    X_ref_train = torch.flatten(X_ref).repeat(X_train.shape[0], 1)
    X_ref_test = torch.flatten(X_ref).repeat(X_test.shape[0], 1)
    train_data_tensor = TensorDataset(X_train, X_ref_train, y_train)
    train_dataloader = DataLoader(train_data_tensor, batch_size=2, shuffle=True)

    test_data_tensor = TensorDataset(X_test, X_ref_test, y_test)
    test_dataloader = DataLoader(test_data_tensor, batch_size=2, shuffle=True)

    return train_dataloader,test_dataloader






