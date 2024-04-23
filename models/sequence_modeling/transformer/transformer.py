import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

#from vocabulary import Vocabulary
from models.sequence_modeling.transformer.encoder import TransformerEncoder
from models.sequence_modeling.transformer.decoder import TransformerDecoder
from models.sequence_modeling.transformer.utils import construct_future_mask
from modules.linear_activation import LinearActivation
from modules.get_activation import get_activation
#from utils import construct_future_mask


class SeqTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: Optional[int]=64,
        ff_dim: Optional[int]=32,
        num_heads: Optional[int]=1,
        num_layers: Optional[int]=1,
        max_decoding_length: Optional[int]=1000,
        vocab_size: Optional[int]=64, # also output size
        padding_idx: Optional[int] = -1,
        bos_idx: Optional[int] = -1,
        dropout_p: Optional[float] = 0.0,
        tie_output_to_embedding: Optional[bool] = None,
        embed:Optional[bool] = False,
        num_classes=0,
        activation_function=LinearActivation,
        activation_kwargs={},
    ):
        super().__init__()
        # Because the encoder embedding, and decoder embedding and decoder pre-softmax transformeation share embeddings
        # weights, initialize one here and pass it on.
        self.embed = None
        if embed:
            self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )
        if type(activation_function) is str:
            activation_function = get_activation(activation_function)
        self.act = activation_function(**activation_kwargs)
        self.fc = None
        if num_classes != 0:
            self.fc = nn.Sequential(nn.Flatten(1), nn.Linear(vocab_size*max_decoding_length, num_classes, bias=False))

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x):
        N,D,T = x.shape      
        x = x.permute([0, 2, 1])  
        encoder_output = self.encoder(x, src_padding_mask=None)
        encoder_output = self.act(encoder_output)
        decoder_input = x
        future_mask = construct_future_mask(seq_len=1).to(x.device)
        decoder_output = self.decoder(decoder_input, encoder_output,src_padding_mask=None, future_mask=future_mask)
        decoder_output = self.act(decoder_output)
        if self.fc is not None:
            return self.fc(decoder_output)
        return decoder_output.permute([0,2,1]).squeeze()



if __name__ == "__main__":
    X = torch.randn(123,53,148)
    model = SeqTransformer(53, 32, 1, 1, 148, 1)
    model2 = SeqTransformer(53, 32, 1, 1, 148, 53, num_classes=8)
    Yh = model(X)
    Yh2 = model2(X)
    print(Yh.shape, Yh2.shape)
