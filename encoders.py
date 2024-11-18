import os
import torch
from torch import nn
import base_model

class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_length, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.angle_rads = self.get_angles()
        self.angle_rads[:, 0::2] = torch.sin(self.angle_rads[:, 0::2])
        self.angle_rads[:, 1::2] = torch.cos(self.angle_rads[:, 1::2])
        self.pos_encoding = self.angle_rads.unsqueeze(0).float()

        
    def get_angles(self):
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        dims = torch.arange(self.embed_dim).unsqueeze(0)
        i = dims//2
        angles = torch.divide(position, torch.pow(1000, (2 * i) / self.embed_dim))
        return angles
    
    def forward(self, x):
        sequence_length = x.size(1)
        pos_encoding = self.pos_encoding[:, :sequence_length, :]
        try:
            return x + pos_encoding.to(x.device)
        except:
            assert False, (x.size(), self.max_seq_length)


class Encoder(base_model.Model):
    def __init__(self, vocab_size, max_seq_length, embed_dim, latent_dim, channels_out=128):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        self.channels_out = channels_out
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.positional_encoder = PositionalEncoder(self.max_seq_length, self.embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, 
                                                       nhead=8, # increased heads
                                                       dim_feedforward=self.embed_dim * 4, # larger feedforward network
                                                       dropout=0.5, # increased dropout
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim), # added batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, x, mask=None):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
            attention_mask = ~mask
            
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer(x, src_key_padding_mask = attention_mask) + x
        x_temp = x.permute(0, 2, 1)
        x_temp = self.conv1(x_temp)
        x_temp = x_temp.permute(0, 2, 1)
        x = x_temp + x
        x = self.projector(x)

        return x


class SMILESEncoder(Encoder):
    def __init__(self, vocab_size, max_seq_length, embed_dim, latent_dim):
        super(SMILESEncoder, self).__init__(vocab_size, max_seq_length, embed_dim, latent_dim)

class ProteinEncoder(Encoder):
    def __init__(self, vocab_size, max_seq_length, embed_dim, latent_dim):
        super(ProteinEncoder, self).__init__(vocab_size, max_seq_length, embed_dim, latent_dim)

if __name__ == "__main__":
    # Test the postional encoder
    # Parameters
    batch_size = 2
    seq_length = 5
    embed_dim = 8
    max_seq_length = 10

    # Create a dummy input tensor (batch_size, seq_length, embed_dim)
    x = torch.rand(batch_size, seq_length, embed_dim)
    pos_encoder = PositionalEncoder(max_seq_length, embed_dim)
    encoded_x = pos_encoder(x)
    print(encoded_x)
    
    # Test SMILES and Proteing encoder
    #Parameters
    vocab_size = 40
    max_seq_length = 512
    embed_dim = 128
    latent_dim = 64
    
    smiles_encoder = SMILESEncoder(vocab_size, max_seq_length, embed_dim, latent_dim)
    protein_encoder = ProteinEncoder(vocab_size, max_seq_length, embed_dim, latent_dim)
    
    print(smiles_encoder)
    print(protein_encoder)
    
    
    