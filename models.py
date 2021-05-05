import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.fc4 = nn.Linear(128, latent_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_mu = self.fc3(x)
        z_logvar = self.fc4(x)
        return z_mu, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=1, output_dim = 50):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self,z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x = F.relu(self.fc3(z))
        return x

class VAE(nn.Module):
    def __init__(self, number_of_genes, number_of_samples,number_of_topics, latent_dim=2): #samples mean spots/cells
        super(VAE, self).__init__()
        self.encoder = Encoder(number_of_genes*number_of_samples, latent_dim)
        self.decoder1 = Decoder(latent_dim//2, number_of_topics*number_of_samples) # Decoder's latent dim should be latent_dim/2
        self.decoder2 = Decoder(latent_dim//2, number_of_genes*number_of_topics)
        self.number_of_samples = number_of_samples
        self.number_of_genes = number_of_genes
        self.number_of_topics = number_of_topics

    def reparametrize(self, mu, logvar):  # Not called in autoencoder approach
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self,x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        z1 = z[:,0].unsqueeze(1)
        z2 = z[:,1].unsqueeze(1)
        x1 = self.decoder1(z1)
        x2 = self.decoder2(z2)
        sample_topic = x1.view(-1,self.number_of_samples, self.number_of_topics)
        gene_topic = x2.view(-1, self.number_of_topics, self.number_of_genes)

        base = torch.matmul(sample_topic,gene_topic)
        output = torch.flatten(base).view(-1,self.number_of_genes*self.number_of_samples)

        return output, gene_topic, sample_topic, z_mu, z_logvar

class STDeconv(nn.Module):
    def __init__(self,number_of_genes, number_of_spots, number_of_cells, topic):
        super(STDeconv, self).__init__()
        self.VAE1 = VAE(number_of_genes=number_of_genes, number_of_samples=number_of_spots, number_of_topics=topic)
        self.VAE2 = VAE(number_of_genes=number_of_genes, number_of_samples=number_of_cells, number_of_topics=topic)

    def forward(self,STData, ScRNASeqData):
        reconST, gene_topicST, spot_topic, STZ_mu, STZ_logvar = self.VAE1(STData)
        reconSC, gene_topicSC, cell_topic, SCZ_mu, SCZ_logvar = self.VAE2(ScRNASeqData)
        return reconST, gene_topicST, spot_topic, reconSC, gene_topicSC, cell_topic, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar