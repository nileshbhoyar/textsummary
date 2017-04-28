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
import pandas as pd
from evaluations import score
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
tf.app.flags.DEFINE_integer("epochs" ,50,"Number of epochs")
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
     
     
     if FLAGS.celltype == 'GRU':
            if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/GRU/noAttention/'
            else:
                ckpt_paths = 'ckpt/checkpoint/GRU/Attention'
            model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
                               num_layers=FLAGS.num_layers,
                             epochs = FLAGS.epochs,
                                lr = 0.005,
                                attention = FLAGS.attention,
                                celltype = 'GRU'
                               )
     else:
             print "graph building started with LSTM Cell"    
             if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/LSTM/noAttention/'
             else:
                ckpt_paths = 'ckpt/checkpoint/LSTM/Attention'
             model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
                              num_layers=FLAGS.num_layers,
                             epochs = FLAGS.epochs,
                                lr = 0.005,
                                attention = FLAGS.attention,
                                celltype = 'LSTM'
                               )

     return model

def self_test():
    print " In Test Mode"  
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
   
    
    model = create_model(metadata,trainX.shape[-1],trainY.shape[-1])
    
    if FLAGS.celltype == 'GRU':
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/GRU/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/GRU/Attention/'
           
    else:
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/LSTM/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/LSTM/Attention/'
           
    print "Retrieving Last Model State"
    XX = np.load('datasets/test_review.npy', mmap_mode='r')
    YY =  np.load('datasets/test_summary.npy', mmap_mode='r')
    result = [[0 for x in range(6)] for y in range(XX.shape[0])] 
   

        
    sess = model.restore_last_session()
    batch_size = 16
    if sess:
         for i in range(0, XX.shape[0], batch_size):
                if (i+1) + batch_size < XX.shape[0]:
                    output = model.predict(sess, XX[i : (i+1) + batch_size ].T)
                    nn = XX[i : (i+1) + batch_size ]
                    for j in range(nn.shape[0]):
                        result[i+j][0] = helpers.decode(sequence=XX[i+j], lookup=metadata['idx2w'], separator=' ')
                        result[i+j][1] = helpers.decode(sequence=YY[i+j], lookup=metadata['idx2w'], separator=' ')
                        
                        result[i+j][2] = helpers.decode(sequence=output.T[j], lookup=metadata['idx2w'], separator=' ')
                        
                        if len(result[i+j][2] ) == 0:
                            result[i+j][2] = ['UNK']
                        if len(result[i+j][1]) != 0:
                              result[i+j][3] =  score.rouge_n(result[i+j][2], result[i+j][1], 1)
                              result[i+j][4] =  score.bleu(result[i+j][2], result[i+j][1], 1)
                              result[i+j][5] = score.f1(result[i+j][3], result[i+j][4])
                        else:
                              result[i+j][3] = result[i+j][4] = result[i+j][5] = 0
    df = pd.DataFrame(result)
    df.columns = ["Review","Actual Summary", "Generated Summary","Rogue1" ,"Bleu1" , "F1"]
    df = df[:-batch_size]
    print ("Average Rogue-1 = %.3f, Max Rouge-1 =%.3f,Min Rogue-1 = %.3f" %(df["Rogue1"].mean(),
                                                                            df["Rogue1"].max(),
                                                                              df["Rogue1"].min()
                                                                            ))
    print ("Average Bleu1 = %.3f, Max Bleu1=%.3f,Min Bleu1 = %.3f" %(df["Bleu1"].mean(),
                                                                            df["Bleu1"].max(),
                                                                              df["Bleu1"].min()
                                                                            ))
    print ("Average F1 = %.3f, Max F1=%.3f,Min F1 = %.3f" %(df["F1"].mean(),
                                                                            df["F1"].max(),
                                                                              df["F1"].min()
                                                                            ))
    result_file = 'results/default.csv'
    if FLAGS.celltype == 'GRU':
        if FLAGS.attention == False: 
                result_file = 'results/GRU_noAttention.csv'
        else:
                result_file = 'results/GRU_Attention.csv'
           
    else:
        if FLAGS.attention == False: 
                result_file = 'results/LSTM_noAttention.csv'
        else:
                result_file = 'results/LSTM_Attention'
    df.to_csv(result_file)             
def decode():
    print "This is for interactive Version....."
    
    
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
   
    
    model = create_model(metadata,trainX.shape[-1],trainY.shape[-1])
    
  
    sess = model.restore_last_session()
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
    #process input strings now"
            inputs = data_utils.ourmodel.data_util.get_tokens(sentence)
            fqtokens =  [w for w in inputs if not w in stopwords.words('english')]
            processed_input = data_utils.ourmodel.data_util.zero_pad_single(fqtokens,metadata['w2idx'])
            #sess = model.restore_last_session()
            output = model.predict(sess, processed_input.T)
            #replies = []

            for ii, ot in zip(processed_input,output.T):
                q = helpers.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
                decoded = helpers.decode(sequence=ot, lookup=metadata['idx2w'], separator=' ').split(' ')
           
    #if decoded.count('unk') == 0:
     #   if decoded not in replies:
                print('Review : [{0}]; Summary : [{1}]'.format(q, ' '.join(decoded)))
           
            sys.stdout.flush()
            sentence = sys.stdin.readline( )  
    #print "Real Summary %s",(helpers.decode(sequence=ot, lookup=metadata['idx2w'], separator=' ').split(' '))
     
        
def train():
    print "Training started ...."
    #metadata = data_utils.ourmodel.data_util.load_metadata()
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
   
    
    model = create_model(metadata,trainX.shape[-1],trainY.shape[-1])
    
    if FLAGS.celltype == 'GRU':
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/GRU/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/GRU/Attention'
           
    else:
        if FLAGS.attention == False: 
                ckpt_paths = 'ckpt/checkpoint/LSTM/noAttention/'
        else:
                ckpt_paths = 'ckpt/checkpoint/LSTM/Attention'
           
    print "Check if model exist already to retrieve"
  
    ckpt = tf.train.get_checkpoint_state(ckpt_paths)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        
        sess = model.restore_last_session()
    else:
        
        print("Created model with fresh parameters.")
        batch_size = FLAGS.batch_size
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