import numpy as np
import csv
import torch
from dtw import dtw
from tqdm import tqdm

datafile = np.load('data/PEMS04/PEMS04.npz')
data=datafile['data']
data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
data_mean = data_mean.squeeze().T 
dtw_distance = np.zeros((data_mean.shape[0], data_mean.shape[0]))
def dist(x, y):
    return np.linalg.norm(x - y)
def manhattan_distance(x, y):
    return np.abs(x - y).sum()
for i in tqdm(range(data_mean.shape[0])):
    for j in range(i, data_mean.shape[0]):
        dtw_distance[i][j] = dtw(data_mean[i], data_mean[j],manhattan_distance)[0]

for i in range(170):
    for j in range(i):
        dtw_distance[i][j] = dtw_distance[j][i]

np.save('dtwpems04.npy',dtw_distance)