import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import bittensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(self.device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(self.device)
        self.encoder = nn.Embedding(ntoken, ninp).to(self.device)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken).to(self.device)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def encode(self, src): 
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        encodings = self.transformer_encoder(src, self.src_mask)
        return encodings

    def decode(self, encodings):
        return self.decoder(encodings)

    def forward(self, sequences):
        encoding = self.encode(sequences)
        output = self.decoder(encoding)
        return output
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
