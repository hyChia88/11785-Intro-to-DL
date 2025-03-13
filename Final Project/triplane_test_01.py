import torch
import torch.nn as nn
import torch.nn.functional as F

class TriPlaneEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(TriPlaneEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.fc_mu = nn.Linear(256 * 8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class TriPlaneDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(TriPlaneDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8 * 8)
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class TriPlaneVAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(TriPlaneVAE, self).__init__()
        self.encoder = TriPlaneEncoder(in_channels, latent_dim)
        self.decoder = TriPlaneDecoder(latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence
