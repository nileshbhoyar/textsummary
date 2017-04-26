#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 07:06:15 2017

@author: nileshbhoyar
"""
import math
import os
import random
import sys
import time
import tensorflow as tf
import numpy as np
import models
import helpers
import data_utils

from models import seq2seq_wrapper
from nltk.corpus import stopwords
import sys
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("celltype","LSTM","Default cell type settings")
tf.app.flags.DEFINE_boolean("attention",False,"No attention by default")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs" ,500,"Number of epochs")
tf.app.flags.DEFINE_integer("num_layers" ,3,"Number of epochs")
FLAGS = tf.app.flags.FLAGS


def create_model(metadata,xseq_len,yseq_len):
     #metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
     #(trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
    # parameters 
     xseq_len = xseq_len
     yseq_len = yseq_len
     
     xvocab_size = len(metadata['idx2w'])  
     yvocab_size = xvocab_size
     emb_dim = 64
<<<<<<< HEAD
     
     
     if FLAGS.celltype == 'GRU':
            if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/GRU/noAttention/'
            else:
                ckpt_paths = 'ckpt/checkpoint/GRU/Attention'
=======
     print "shapes " ,xseq_len,yseq_len
     ckpt_paths = 'ckpt/checkpoint/GRU/'
     if FLAGS.celltype == 'GRU':
            print "Create model with GRU Cell"
            ckpt_paths = 'ckpt/checkpoint/GRU/'
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
            model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
<<<<<<< HEAD
                               num_layers=2,
                             epochs = 50,
                                lr = 0.005,
                                attention = FLAGS.attention,
=======
                               num_layers=3,
                             epochs = 5000,
                                lr = 0.002,
                                attention = True,
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
                                celltype = 'GRU'
                               )
     else:
             print "graph building started with LSTM Cell"    
<<<<<<< HEAD
             if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/LSTM/noAttention/'
             else:
                ckpt_paths = 'ckpt/checkpoint/LSTM/Attention'
=======
             ckpt_paths = 'ckpt/checkpoint/LSTM/'
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
             model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
<<<<<<< HEAD
                               num_layers=2,
                             epochs = 50,
                                lr = 0.005,
                                attention = FLAGS.attention,
=======
                               num_layers=3,
                             epochs = 5000,
                                lr = 0.002,
                                attention = True,
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
                                celltype = 'LSTM'
                               )

     return model

def self_test():
    print "I am in self test :this part if to -do"
def decode():
    print "This is for interactive Version....."
<<<<<<< HEAD
    
    
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
   
    
    model = create_model(metadata,trainX.shape[-1],trainY.shape[-1])
    
  
=======
   # print "Training started ...."
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()

    model = create_model()
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
    sess = model.restore_last_session()
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
    #process input strings now"
            inputs = data_utils.ourmodel.data_util.get_tokens(sentence)
            fqtokens =  [w for w in inputs if not w in stopwords.words('english')]
            processed_input = data_utils.ourmodel.data_util.zero_pad_single(fqtokens,metadata['w2idx'])
<<<<<<< HEAD
            #sess = model.restore_last_session()
            output = model.predict(sess, processed_input.T)
            #replies = []

            for ii, ot in zip(processed_input,output.T):
=======
            
            output = model.predict(sess, processed_input.T)
           

            for ii, oi in zip(processed_input.T, output):
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
                q = helpers.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
                decoded = helpers.decode(sequence=ot, lookup=metadata['idx2w'], separator=' ').split(' ')
           
    #if decoded.count('unk') == 0:
     #   if decoded not in replies:
                print('Review : [{0}]; Summary : [{1}]'.format(q, ' '.join(decoded)))
           
            sys.stdout.flush()
            sys.stdout.write("Input statement :")
            sentence = sys.stdin.readline( )  
    #print "Real Summary %s",(helpers.decode(sequence=ot, lookup=metadata['idx2w'], separator=' ').split(' '))
     
        
def train():
    print "Training started ...."
    #metadata = data_utils.ourmodel.data_util.load_metadata()
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
   
    
    model = create_model(metadata,trainX.shape[-1],trainY.shape[-1])
    
    if FLAGS.celltype == 'GRU':
<<<<<<< HEAD
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/GRU/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/GRU/Attention'
           
    else:
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/LSTM/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/LSTM/Attention'
           
=======
            ckpt_paths = 'ckpt/checkpoint/GRU/'
    else:
            ckpt_paths = 'ckpt/checkpoint/LSTM/'
>>>>>>> add20b511d02418c7637260abb87e9b0bc584fd4
    print "Check if model exist already to retrieve"
  
    ckpt = tf.train.get_checkpoint_state(ckpt_paths)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        
        sess = model.restore_last_session()
    else:
        
        print("Created model with fresh parameters.")
        batch_size = 16
        #val_batch_gen = helpers.rand_batch_gen(validX, validY, batch_size)
        #train_batch_gen = helpers.rand_batch_gen(trainX, trainY, batch_size)
        #sess =  model.train(train_batch_gen, val_batch_gen)
        sess =  model.train_batch_file(batch_size = batch_size)
    
    
    print "Training Complete"

def main(_):
  print FLAGS
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
