import torch
from models import STDeconv
from loss_fn import totalloss
import numpy as np
from evaluation import SpotCellEval

def save_ckp(state, f_path='./best_model.pt'):
    torch.save(state, f_path)


def train_network(learning_rate=0.0001, num_epochs=200, batch_size=2, n_genes = 1000, n_spots=200, n_cells=600, n_topic=200, train_data_loader=None, test_data_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STDeconv(number_of_genes=n_genes, number_of_spots=n_spots, number_of_cells=n_cells, topic=n_topic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_accuracy = []
    epoch_valid_accuracy = []
    min_loss = np.inf
    for epoch in range(num_epochs):
        train_loss_val = 0.0
        model.train()
        print("Epoch {}".format(epoch + 1))
        for batch_idx, Mats in enumerate(train_data_loader):
            STMat = Mats[0]
            SCMat = Mats[1]
            STMat = STMat.to(device=device).reshape(batch_size,-1)
            SCMat = SCMat.to(device=device).reshape(batch_size,-1)
            reconST, gene_topicST, spot_topic, reconSC, gene_topicSC, cell_topic, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar = model(STMat, SCMat)

            loss = totalloss(STMat, reconST, SCMat, reconSC, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar, gene_topicST, gene_topicSC,beta=10)
            loss.backward()
            optimizer.step()
            train_loss_val +=  loss.item()
            train_accuracy,_,_ = SpotCellEval(spot_topic, cell_topic).test_eval(Mats[2])
        epoch_train_loss.append(train_loss_val / len(train_data_loader))
        epoch_train_accuracy.append(train_accuracy/len(train_data_loader))
        print('Epoch {}, Training Loss {:.6f}'.format(epoch, epoch_train_loss[epoch]))
        print('Epoch {}, Training Accuracy {:.6f}'.format(epoch, epoch_train_accuracy[epoch]))

        model.eval()
        valid_loss_val =0
        for batch_idx, Mats in enumerate(test_data_loader):
            STMat = Mats[0]
            SCMat = Mats[1]
            STMat = STMat.to(device=device).reshape(batch_size,-1)
            SCMat = SCMat.to(device=device).reshape(batch_size,-1)
            reconST, gene_topicST, spot_topic, reconSC, gene_topicSC, cell_topic, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar = model(STMat, SCMat)

            loss = totalloss(STMat, reconST, SCMat, reconSC, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar, gene_topicST, gene_topicSC,beta=10)
            loss.backward()
            optimizer.step()
            valid_loss_val +=  loss.item()
            valid_accuracy,_,_ = SpotCellEval(spot_topic, cell_topic).test_eval(Mats[2])

        epoch_valid_loss.append(valid_loss_val / len(test_data_loader))
        if epoch_valid_loss[epoch]< min_loss:
            print('Validation Loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss,epoch_valid_loss[epoch]))
            # save checkpoint as best model

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': epoch_valid_loss[epoch],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_ckp(checkpoint, './best_model_Topic_{}.pt'.format(n_topic))
            min_loss = epoch_valid_loss[epoch]
        print("Epoch {} Validation Loss {:.6f}".format(epoch,epoch_valid_loss[epoch]))
        print('Epoch {} Validation Accuracy {:.6f}'.format(epoch, epoch_valid_accuracy[epoch]))

if __name__=="__main__":
    train_network()