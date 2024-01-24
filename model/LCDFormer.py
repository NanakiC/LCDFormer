from torch.nn import BatchNorm2d, Conv2d, ModuleList
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from tensorboardX import SummaryWriter
import copy
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
import copy
import math
from scipy import sparse as sp
from utils.utils import clones
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
class TCN(nn.Module):
    def __init__(self, dropout=0.1, in_dim=1, hid_dim=64,cnn_layers=4, group=4):
        super(TCN, self).__init__()
        self.start_conv = Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.cnn_layers = cnn_layers
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        self.group=group
        D = [1, 2, 4, 8]
        additional_scope = 1
        receptive_field = 1
        for i in range(self.cnn_layers):
            self.filter_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            self.gate_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field=receptive_field
        depth = list(range(self.cnn_layers))
        self.bn = ModuleList([BatchNorm2d(hid_dim) for _ in depth])
        self.se=SELayer(hid_dim)


    def forward(self, input):#B,N,T,F
        input=input.permute(0,3,1,2) #B,F,N,T 4,1,307,12
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0)) #4,1,307,16
        x = self.start_conv(input) #4,64,307,16
        skip = 0
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            if self.group !=1:
                x = channel_shuffle (x,self.group)
            try:
                skip += x[:, :, :, -1:]
            except:
                skip = 0
            if i == self.cnn_layers-1:
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = torch.squeeze(skip, dim=-1) #4,64,307
        x=self.se(x)
        x=x.unsqueeze(-1) #4,64,307,1
        x = x.permute(0,2,3,1)
        return x
class TCNs(nn.Module):
    def __init__(self, dropout=0.1, in_dim=1, hid_dim=64,cnn_layers=4, group=4):
        super(TCNs, self).__init__()
        self.tcns = clones(TCN(), 12)
    def forward(self, input):#4,307,144,1
        for i in range(12):
            x=input[:,:,i*12:(i+1)*12,:] #4,307,12,1
            output=self.tcns[i](x)
            if i==0:
                outputs=output
            else:
                outputs = torch.cat((outputs, output), dim=2)
        return outputs
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, src, trg):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        encoder_output = self.encode(src)  # (batch_size, N, T_in, d_model)

        return self.decode(trg, encoder_output)

    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        h = self.src_embed(src)
        return self.encoder(h)
        # return self.encoder(self.src_embed(src))

    def decode(self, trg, encoder_output):
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))
def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)

        return self.norm(x)
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
#             x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m))
            return self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)  # output: (batch, N, T', d_model)
#             x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)

class Encoder(nn.Module):
    def __init__(self, layer,N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)
class SpatialPositionalEncoding(nn.Module):
    def __init__(self,lap_pos_enc, d_model, num_of_vertices, dropout):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.lap_pos_enc=lap_pos_enc
        self.embedding_lap_pos_enc = nn.Linear(d_model, d_model)
    def forward(self, x):
        
        batch, num_of_vertices, timestamps, _ = x.shape
        h_lap_pos_enc=torch.Tensor(self.lap_pos_enc).to(x.device)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc).unsqueeze(0)# (N, d_model)->(1,N,d_model)

        x = x + h_lap_pos_enc.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())
class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):  # key causal; query causal;
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        '''
        :param nb_head:
        :param d_model:
        :param num_of_weeks:
        :param num_of_days:
        :param num_of_hours:
        :param points_per_hour:
        :param kernel_size:
        :param dropout:
        '''
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

       
        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)
def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  
class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)
                #l,x参与循环，x第一次循环表示query 第二次表示key ,l表示两次循环的二维卷积
            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)
def Spatial_Attention_layer(query, key):
    '''

    :param query:  (batch, T, h, N, d_k)
    :param key: (batch, T, h, N, d_k)
    :return: (batch, T, h, N, N)
        '''
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # scores: (batch, T, h, N, N))

    return score  #(batch, T, h, N, N)
class global_attention(nn.Module):
    def __init__(self, in_channels, out_channels,nb_head,dropout=.0):
        super(global_attention, self).__init__()
        self.d_k = in_channels // nb_head
        self.h = nb_head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.linears = clones(nn.Linear(in_channels, in_channels), 3)
    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        x=x.permute(0,2,1,3)#(batch_size, T, N, F_in)
        batch_size,num_of_timesteps,num_of_vertices, in_channels = x.shape
        
        query=self.linears[0](x)
        query=query.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        query=query.transpose(2, 3)#(batch, T, h, N, d_k
        
        key=self.linears[1](x)
        key=key.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        key=key.transpose(2, 3)#(batch, T, h, N, d_k
        
        
        value=self.linears[2](x)
        value=value.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        value=value.transpose(2, 3)#(batch, T, h, N, d_k
        
        spatial_attention = Spatial_Attention_layer(query,key)   # scaled self attention: (batch, T, h,N, N)
        spatial_attention=spatial_attention.softmax(dim=-1)
        x = (spatial_attention @ value) #batch, T, h, N, d_k
        x=x.permute(0,3,1,2,4)
        x=x.reshape(batch_size, num_of_timesteps, num_of_vertices, in_channels)

        return F.relu(x).permute(0,2,1,3)
class adj_dtw_attention(nn.Module):
    def __init__(self, mask_dtw, in_channels, out_channels,nb_head,dropout=.0):
        super(adj_dtw_attention, self).__init__()
        self.d_k = in_channels // nb_head
        self.h = nb_head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.linears = clones(nn.Linear(in_channels, in_channels), 3)
        self.mask_dtw=mask_dtw.cuda()
    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        x=x.permute(0,2,1,3)#(batch_size, T, N, F_in)
        batch_size,num_of_timesteps,num_of_vertices, in_channels = x.shape
        
        query=self.linears[0](x)
        query=query.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        query=query.transpose(2, 3)#(batch, T, h, N, d_k
        
        key=self.linears[1](x)
        key=key.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        key=key.transpose(2, 3)#(batch, T, h, N, d_k
        
        
        value=self.linears[2](x)
        value=value.view(batch_size, num_of_timesteps, -1, self.h, self.d_k)#(batch_size, T, N, h,d_k)
        value=value.transpose(2, 3)#(batch, T, h, N, d_k
        
        spatial_attention = Spatial_Attention_layer(query,key)   # scaled self attention: (batch, T, h,N, N)
        spatial_attention.masked_fill_(self.mask_dtw, float('-inf'))
        spatial_attention=spatial_attention.softmax(dim=-1)
        x = (spatial_attention @ value) #batch, T, h, N, d_k
        x=x.permute(0,3,1,2,4)
        x=x.reshape(batch_size, num_of_timesteps, num_of_vertices, in_channels)

        return F.relu(x).permute(0,2,1,3)

class PositionWiseATTFeedForward(nn.Module):
    def __init__(self, att, dropout=.0):
        super(PositionWiseATTFeedForward, self).__init__()
        self.att = att
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.att(x)))
class GLU(nn.Module): 
    def __init__(self,d_model, kernel_size=3, dropout=.0):
        super(GLU, self).__init__()

        self.linears = nn.Linear(d_model, d_model)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1)//2
        self.cauasl_conv = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.causal_padding))
        self.conv = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))

    def forward(self, causal_out,encoder_out):

        batch_size,num_nodes,T,d_model=causal_out.shape
        causal = self.cauasl_conv(causal_out.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous()#([4, 64, 307, T1])

        con = self.conv(encoder_out.permute(0, 3, 1, 2))[:,:,:,:T].contiguous()#(B,D,N,T1)

        temp=(causal+torch.sigmoid(con)).contiguous()
        out=(F.relu(temp + self.conv1(encoder_out.permute(0, 3, 1, 2))[:,:,:,:T])).contiguous()

        
        
        x = out.permute(0, 2, 3, 1).contiguous()  # (batch, N, T1, h, d_k)  # (batch, N, T1, d_model)
        return self.linears(x)