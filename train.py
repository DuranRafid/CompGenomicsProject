import torch
from models import STDeconv
from loss_fn import totalloss
import numpy as np

def save_ckp(state, f_path='./best_model.pt'):
    torch.save(state, f_path)


def train_network(learning_rate=0.0001, num_epochs=200, batch_size=2, n_genes = 1000, n_spots=200, n_cells=600, n_topic=50,data_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STDeconv(number_of_genes=n_genes, number_of_spots=n_spots, number_of_cells=n_cells, topic=n_topic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_loss = []
    min_loss = np.inf
    for epoch in range(num_epochs):
        loss_val = 0.0
        model.train()
        print("Epoch {}".format(epoch + 1))
        for batch_idx, Mats in enumerate(data_loader):
            STMat = Mats[0]
            SCMat = Mats[1]
            STMat = STMat.to(device=device).reshape(batch_size,-1)
            SCMat = SCMat.to(device=device).reshape(batch_size,-1)
            reconST, gene_topicST, spot_topic, reconSC, gene_topicSC, cell_topic, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar = model(STMat, SCMat)

            loss = totalloss(STMat, reconST, SCMat, reconSC, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar, gene_topicST, gene_topicSC,beta=10)
            loss.backward()
            optimizer.step()
            loss_val +=  loss.item()
        epoch_loss.append(loss_val / batch_size)
        if epoch_loss[epoch]< min_loss:
            print('Loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss,epoch_loss[epoch]))
            # save checkpoint as best model

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': epoch_loss[epoch],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            np.savetxt('Cell_Topic_Mat_Topic{}.txt'.format(n_topic),cell_topic[0].cpu().detach().numpy())
            np.savetxt('Spot_Topic_Mat_Topic{}.txt'.format(n_topic), spot_topic[0].cpu().detach().numpy())
            save_ckp(checkpoint, './best_model_Topic_{}.pt'.format(n_topic))
            min_loss = epoch_loss[epoch]
        print("Loss {}".format(epoch_loss[epoch]))

if __name__=="__main__":
    train_network()