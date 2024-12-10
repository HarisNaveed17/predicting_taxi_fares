import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TimeTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers, output_dim, seq_len):
        super(TimeTransformer, self).__init__()

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.xavier_uniform_(self.positional_encoding)

        # Input embedding
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Add positional encoding to input embeddings
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:seq_len]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Aggregate the last timestep for prediction
        x = x.mean(dim=1)  # Global average pooling

        # Output layer
        return self.fc_out(x)
