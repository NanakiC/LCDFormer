import numpy as np
import os
import torch
import torch.utils.data
import torch.nn as nn
import copy
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  

            with open(distance_df_filename, 'r') as f:
                f.readline()  
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j= int(row[0]), int(row[1])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
            return A

        else:  

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j = int(row[0]), int(row[1])
                    A[i, j] = 1
                    A[j, i] = 1
            return A



def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])