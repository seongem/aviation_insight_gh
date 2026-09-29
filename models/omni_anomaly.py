import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, latent_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(1, latent_dim))

    def forward(self, z):
        # Implementation of Planar Flow: f(z) = z + u * h(w^T * z + b)
        # Ensuring invertibility with the constraint from the paper
        inner = torch.mm(z, self.w.t()) + self.b
        return z + self.u * torch.tanh(inner)

class OmniAnomaly(nn.Module):
    def __init__(self, n_features, window_size, latent_dim=32, rnn_hidden=64, flow_layers=3):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim

        # 1. Encoder: GRU to capture temporal dependencies
        self.encoder_rnn = nn.GRU(n_features, rnn_hidden, batch_first=True)
        self.tr_mu = nn.Linear(rnn_hidden, latent_dim)
        self.tr_logvar = nn.Linear(rnn_hidden, latent_dim)

        # 2. Normalizing Flows (Planar Flows)
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(flow_layers)])

        # 3. Decoder: GRU to reconstruct the sequence
        self.decoder_rnn = nn.GRU(latent_dim, rnn_hidden, batch_first=True)
        self.reconstruction_head = nn.Linear(rnn_hidden, n_features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoding
        _, h_n = self.encoder_rnn(x)  # h_n shape: [1, batch, rnn_hidden]
        h_n = h_n.squeeze(0)
        
        mu = self.tr_mu(h_n)
        logvar = self.tr_logvar(h_n)
        z_0 = self.reparameterize(mu, logvar)

        # Apply Normalizing Flows to transform the latent distribution
        z_k = z_0
        for flow in self.flows:
            z_k = flow(z_k)

        # Decoding: Expand z_k back to the window sequence
        # We repeat the latent vector across the window size for the GRU
        z_sequence = z_k.unsqueeze(1).repeat(1, self.window_size, 1)
        recon_features, _ = self.decoder_rnn(z_sequence)
        x_recon = self.reconstruction_head(recon_features)

        return x_recon, mu, logvar