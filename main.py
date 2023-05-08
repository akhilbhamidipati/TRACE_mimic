import datetime
import numpy as np
import os
import random
import torch

from statsmodels.tsa import stattools
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, average_precision_score
from torch.utils import data

from cpc import cpc
from end_to_end import end_to_end
from trace_model import trace
from triplet_loss import triplet_loss

# Interval between recordings
interval = 1

# Setting hyperparams
encoder_type = "CausalCNNEncoder"

# Learn encoder hyperparams
window_size = 60
w = 0.05
batch_size = 30
lr = .00005
decay = 0.0005
mc_sample_size = 6
n_epochs = 150
data_type = "mimic"
n_cross_val_encoder = 1
ETA = 4
ADF = False
ACF = False
ACF_PLUS = True
ACF_nghd_Threshold = 0.6
ACF_out_nghd_Threshold = 0.1

# CausalCNNEncoder Hyperparameters
CausalCNNEncoder_in_channels = 6
CausalCNNEncoder_channels = 4
CausalCNNEncoder_depth = 1
CausalCNNEncoder_reduced_size = 2
CausalCNNEncoder_encoding_size = 10
CausalCNNEncoder_kernel_size = 2
CausalCNNEncoder_window_size = 12

n_cross_val_classification = 3

encoder_hyper_params = {'in_channels': CausalCNNEncoder_in_channels,
                            'channels': CausalCNNEncoder_channels, 
                            'depth': CausalCNNEncoder_depth, 
                            'reduced_size': CausalCNNEncoder_reduced_size,
                            'encoding_size': CausalCNNEncoder_encoding_size,
                            'kernel_size': CausalCNNEncoder_kernel_size,
                            'window_size': CausalCNNEncoder_window_size}

learn_encoder_hyper_params = {'window_size': window_size,
                                'w': w,
                                'batch_size': batch_size,
                                'lr': lr,
                                'decay': decay,
                                'mc_sample_size': mc_sample_size,
                                'n_epochs': n_epochs,
                                'data_type': data_type,
                                'n_cross_val_encoder': n_cross_val_encoder,
                                'cont': True,
                                'ETA': ETA,
                                'ADF': ADF,
                                'ACF': ACF,
                                'ACF_PLUS': ACF_PLUS,
                                'ACF_nghd_Threshold': ACF_nghd_Threshold,
                                'ACF_out_nghd_Threshold': ACF_out_nghd_Threshold}


classification_hyper_params = {'n_cross_val_classification': n_cross_val_classification}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'device' not in encoder_hyper_params:
        encoder_hyper_params['device'] = device
    
if 'device' not in learn_encoder_hyper_params:
    learn_encoder_hyper_params['device'] = device

pretrain_hyper_params = {}


def main():
    print("ENCODER HYPER PARAMETERS")
    for key in encoder_hyper_params:
        print(key)
        print(encoder_hyper_params[key])
        print()
    print("LEARN ENCODER HYPER PARAMETERS")
    for key in learn_encoder_hyper_params:
        print(key)
        print(learn_encoder_hyper_params[key])
        print()

    print("Executing TRACE model ...")
    trace(data_type, encoder_type, encoder_hyper_params, learn_encoder_hyper_params, classification_hyper_params, pretrain_hyper_params)
    print("TRACE model finished")
    print()

    print("Executing CPC model for comparison ...")
    cpc(data_type, lr=0.0001, cv=1)
    print("CPC model finished")
    print()

    print("Executing Triplet-Loss model for comparison ...")
    triplet_loss(data_type, 0.0001, 1)
    print("Triplet-Loss model finished")
    print()

    print("Executing End-to-End model for comparison ...")
    end_to_end(data_type, 3)
    print("End-to-End model finished")
    print()


if __name__ == "__main__":
     main()