import torch
import torch.nn.functional as F


def vae_loss(input_mat, output_mat, mu, logvar, beta=10):
    n= input_mat.size(0)

    recon_loss = F.mse_loss(input_mat, output_mat, reduction="mean").div(n)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return recon_loss + kl*beta

def ST_SCDeconvloss(gene_topicST, gene_topicSC):
    n = gene_topicST.size(0)
    loss = F.mse_loss(gene_topicSC, gene_topicSC,reduction="mean").div(n)
    return loss

def totalloss(STMat, STMat_Recon, SCMat, SCMat_Recon, STZ_mu, STZ_logvar, SCZ_mu, SCZ_logvar, gene_topicST, gene_topicSC,beta=10):
    STloss = vae_loss(STMat, STMat_Recon, STZ_mu, STZ_logvar,beta)
    SCloss = vae_loss(SCMat, SCMat_Recon, SCZ_mu, SCZ_logvar,beta)
    SCSTloss = ST_SCDeconvloss(gene_topicST, gene_topicSC)
    total_loss = STloss + SCloss + SCSTloss
    return total_loss
