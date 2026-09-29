import torch
import torch.nn as nn


class LSTMVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(LSTMVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder Network
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder Network
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick: z = mu + std * eps"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # Encode: extract final hidden state
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # Take top layer hidden state: [batch_size, hidden_dim]

        # Latent distributions
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]

        # Decoder: project z and repeat across sequence dimension
        z_projected = self.decoder_input(z)  # [batch_size, hidden_dim]
        decoder_in = z_projected.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]

        # Decode sequence
        dec_out, _ = self.decoder_lstm(decoder_in)
        x_recon = self.fc_out(dec_out)  # [batch_size, seq_len, input_dim]

        return x_recon, mu, logvar


def lstm_vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Computes MSE Reconstruction Loss + KL Divergence Penalty."""
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return recon_loss + beta * kl_div