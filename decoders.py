import os
import torch
import encoders
import base_model
from torch import nn

class Decoder(base_model.Model):
    def __init__(self, vocab_size, max_seq_length, embed_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.positional_encoder = encoders.PositionalEncoder(self.max_seq_length, self.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, nhead=8, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, input_, mem):
        mem = self.fc(mem)
        x = self.embedding(input_)
        x = self.positional_encoder(x)
        x = self.transformer(x, mem) + x
        x = self.dropout(x)
        return x


class SMILESDecoder(Decoder):
    def __init__(self, vocab_size, max_seq_length, embed_dim):
        super(SMILESDecoder, self).__init__(vocab_size, max_seq_length, embed_dim)

class ProteinDecoder(Decoder):
    def __init__(self, vocab_size, max_seq_length, embed_dim):
        super(ProteinDecoder, self).__init__(vocab_size, max_seq_length, embed_dim)
        
        