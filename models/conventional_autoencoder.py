import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape: tuple of (seq_len, num_features) e.g., (128, 36)
        """
        super(ConvAutoencoder, self).__init__()
        self.seq_len, self.num_features = input_shape

        # Encoder: (Batch, Features, Steps)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True), # Matches Keras padding='same' for maxpool
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        )

        # Decoder: (Batch, Channels, Steps)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=32, out_channels=self.num_features, kernel_size=3, padding=1)
            # Linear activation (default output)
        )

    def forward(self, x):
        # x input shape: (Batch, Sequence_Length, Features)
        # Permute to PyTorch standard: (Batch, Features, Sequence_Length)
        x = x.permute(0, 2, 1)

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        # Permute back to: (Batch, Sequence_Length, Features)
        reconstructed = reconstructed.permute(0, 2, 1)

        # Ensure output sequence length matches input sequence length
        if reconstructed.shape[1] != self.seq_len:
            reconstructed = reconstructed[:, :self.seq_len, :]

        return reconstructed