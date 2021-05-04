import torch
from models import STDeconv
from loss_fn import totalloss

def train_network(learning_rate=0.0001, num_epochs=200, batch_size=2, data_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STDeconv(number_of_genes=1000, number_of_spots=200, number_of_cells=600, topic=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_loss = []
    for epoch in range(num_epochs):
        loss_val = 0.0
        model.train()
        print("Epoch {}".format(epoch + 1))
        for batch_idx, STMat, SCMat in enumerate(data_loader):
            STMat = STMat.to(device=device).reshape(batch_size,-1)
            SCMat = SCMat.to(device=device).reshape(batch_size,-1)
            reconST, gene_topicST, spot_topic, reconSC, gene_topicSC, cell_topic, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar = model(STMat, SCMat)
            loss = totalloss(STMat, reconST, SCMat, reconST, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar, gene_topicST, gene_topicSC,beta=10)
            loss.backward()
            optimizer.step()
            loss_val +=  loss.item()
        epoch_loss.append(loss_val / batch_size)
        print("Loss {}".format(epoch_loss[epoch]))

if __name__=="__main__":
    train_network()