#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 07:06:15 2017

@author: nileshbhoyar
"""

import tensorflow as tf
import numpy as np
import models
import helpers
import data_utils
metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
(trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)


# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 200

from models import seq2seq_wrapper
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/checkpoint/',
                               emb_dim=emb_dim,
                               num_layers=3,
                             epochs = 1000,
                                lr = 0.05
                               )