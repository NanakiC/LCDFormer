#!/usr/bin/env python
# coding: utf-8
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
from utils.utils import *
from utils.get import *
from model.LCDFormer import *


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
  
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # 

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 
            decoder_input_list = [decoder_start_inputs]

            # 
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'outputepoch%s' % (epoch))
#         np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type):
    

    params_filename = os.path.join(params_path, 'epoch%s' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type)
def compute_val_loss(net,val_loader,criterion,epoch):
   

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B，N，T，1)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]
            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        #sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss
def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']#大小归一化

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])
# (10181, 307, 12)
    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)
    #decoder输入：训练集x最后一个+训练集y前11个
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch']) 
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])
filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])

use_LayerNorm = True
residual_connection = True

adj_mx= get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

folder_dir = '%s_h%dd%dw%d' % (model_name, num_of_hours, num_of_days, num_of_weeks)
print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('../experiments', dataset_name, folder_dir)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_graphdata_normY_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)


A=adj_mx.copy()
lap_pos_enc=get_lap_pos_enc(A,d_model)
c = copy.deepcopy
adj_dtw=np.load('dtwpems04.npy')
mask_mx=get_mx_mask(adj_mx,num_of_vertices)
mask_dtw=get_dtw_mask(adj_dtw,num_of_vertices)
src_dense=TCNs()
max_len = num_for_predict
en_lookup_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
position_wise_global = PositionWiseATTFeedForward(global_attention(d_model, d_model,nb_head), dropout=dropout)
position_wise_dtw = PositionWiseATTFeedForward(adj_dtw_attention(mask_dtw, d_model, d_model,nb_head), dropout=dropout)
trg_dense = nn.Linear(1, d_model)
attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, 0, 0, 1, num_for_predict, kernel_size, dropout=dropout)
att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model,0, 0, 1, num_for_predict, kernel_size, dropout=dropout)
attn_st = GLU(64)
encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)
decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
spatial_position = SpatialPositionalEncoding(lap_pos_enc,d_model, num_of_vertices, dropout)
encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
encoderLayer=EncoderLayer(d_model, attn_ss,c(position_wise_dtw), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
encoder = Encoder(encoderLayer, num_layers)
decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_global), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
decoder = Decoder(decoderLayer, num_layers)
generator = nn.Linear(d_model, 1)
net = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)
if (start_epoch == 0) and (not os.path.exists(params_path)): 
    os.makedirs(params_path)
    print('create params directory %s' % (params_path), flush=True)
elif (start_epoch == 0) and (os.path.exists(params_path)):
    shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('delete the old one and create params directory %s' % (params_path), flush=True)
elif (start_epoch > 0) and (os.path.exists(params_path)):  
    print('train from params directory %s' % (params_path), flush=True)
else:
    raise SystemExit('Wrong type of model!')
criterion = nn.L1Loss().to(DEVICE)  
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
global_step = 0
best_epoch = 0
best_val_loss = np.inf
if start_epoch > 0:

    params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

    net.load_state_dict(torch.load(params_filename))

    print('start epoch:', start_epoch, flush=True)

    print('load weight from: ', params_filename, flush=True)
start_time = time()
for epoch in range(start_epoch, epochs):
    params_filename = os.path.join(params_path, 'epoch%s' % epoch)
    val_loss = compute_val_loss(net, val_loader, criterion, epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename, flush=True)
    net.train()  # ensure dropout layers are in train mode
    train_start_time = time()
    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, decoder_inputs, labels = batch_data
        #encoder_inputs: B,N,F,T
        #decoder_inputs: B,N,T
        #labels:B,N,T
        encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
        decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
        labels = labels.unsqueeze(-1) # (B, N, T, 1)
        optimizer.zero_grad()
        outputs = net(encoder_inputs, decoder_inputs) #B,N,T,F
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        global_step += 1
#     epoch_scheduler.step()
    print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
    print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)
print('best epoch:', best_epoch, flush=True)
print('apply the best val model on the test data set ...', flush=True)
predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')


optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
print('fine tune the model ... ', flush=True)
for epoch in range(epochs, epochs+fine_tune_epochs):
    params_filename = os.path.join(params_path, 'epoch%s' % epoch)
    net.train()  # ensure dropout layers are in train mode
    train_start_time = time()
    for batch_index, batch_data in enumerate(train_loader):
        #encoder_inputs: B,N,F,T
        #decoder_inputs: B,N,T
        #labels:B,N,T
        encoder_inputs, decoder_inputs, labels = batch_data
        encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
        decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
        labels = labels.unsqueeze(-1) # (B, N, T, 1)
        predict_length = labels.shape[2]  # T
        optimizer.zero_grad()
        encoder_output = net.encode(encoder_inputs)
        # decode
        decoder_start_inputs = decoder_inputs[:, :, :1, :]
        decoder_input_list = [decoder_start_inputs]
        for step in range(predict_length):
            decoder_inputs = torch.cat(decoder_input_list, dim=2)
            predict_output = net.decode(decoder_inputs, encoder_output)
            decoder_input_list = [decoder_start_inputs, predict_output]
        loss = criterion(predict_output, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        global_step += 1
    print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
    print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)
    val_loss = compute_val_loss(net, val_loader, criterion,epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename, flush=True)
print('best epoch:', best_epoch, flush=True)
print('apply the best val model on the test data set ...', flush=True)
predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')

