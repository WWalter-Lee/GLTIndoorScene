import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff, dropout, activation):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation
    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e9)

    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) #
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        self.global_attn = MultiHeadAttention(heads, d_model)
        self.ref_attn = MultiHeadAttention(heads, d_model)
        self.glb_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_model * 2, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ref_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout(self.ref_attn(x2, x2, x2, ref_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, dropout, activation, layer_num):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout, activation) for _ in range(layer_num)])
    def forward(self, x, ref_mask):
        for layer in self.layers:
            x = layer(x, ref_mask)
        return x