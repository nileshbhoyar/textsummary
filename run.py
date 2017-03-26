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
from models import seq2seq_wrapper
from nltk.corpus import stopwords
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("celltype","LSTM","Default cell type settings")
tf.app.flags.DEFINE_boolean("attention",False,"No attention by default")
tf.app.flags.DEFINE_integer("batch_size", 2,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs" ,30,"Number of epochs")
tf.app.flags.DEFINE_integer("num_layers" ,3,"Number of epochs")
FLAGS = tf.app.flags.FLAGS


def create_model():
     metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
     (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)
    # parameters 
     xseq_len = trainX.shape[-1]
     yseq_len = trainY.shape[-1]
     batch_size = FLAGS.batch_size
     xvocab_size = len(metadata['idx2w'])  
     yvocab_size = xvocab_size
     emb_dim = 64
     
     ckpt_paths = 'ckpt/checkpoint/GRU'
     if FLAGS.celltype == 'GRU':
            ckpt_paths = 'ckpt/checkpoint/GRU'
            model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
                               num_layers=3,
                             epochs = 50,
                                lr = 0.02,
                                attention = True,
                                celltype = 'GRU'
                               )
     else:
             print "graph building started with LSTM Cell"    
             ckpt_paths = 'ckpt/checkpoint/LSTM'
             model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt_paths,
                               emb_dim=emb_dim,
                               num_layers=3,
                             epochs = 50,
                                lr = 0.02,
                                attention = True,
                                celltype = 'LSTM'
                               )

     return model

def self_test():
    print "I am in self test :this part if to -do"
def decode():
    print "This is for interactive Version....."
    print "Training started ...."
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    with tf.Session() as sess:
        model = create_model()
        sess = model.restore_last_session()
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
    #process input strings now"
            inputs = data_utils.ourmodel.data_util.get_tokens(sentence)
            fqtokens =  [w for w in qtokens if not w in stopwords.words('english')]
            processed_input = zdata_utils.ourmodel.data_util.zero_pad_single(inputs,metadata['w2idx'])
            sess = model.restore_last_session()
            output = model.predict(sess, input_)
            replies = []

            for ii, oi,ot in zip(input_.T, output,output_.T):
                q = helpers.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
                decoded = helpers.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
           
    #if decoded.count('unk') == 0:
     #   if decoded not in replies:
                print('Review : [{0}]; Summary : [{1}]'.format(q, ' '.join(decoded)))
           
            sys.stdout.flush()
            sentence = sys.stdin.readline( )  
    #print "Real Summary %s",(helpers.decode(sequence=ot, lookup=metadata['idx2w'], separator=' ').split(' '))
     
        
def train():
    print "Training started ...."
    metadata, idx_q, idx_a = data_utils.ourmodel.data_util.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = helpers.split_dataset(idx_q, idx_a)

    
    model = create_model()
    if FLAGS.celltype == 'GRU':
            ckpt_paths = 'ckpt/checkpoint/GRU'
    else:
            ckpt_paths = 'ckpt/checkpoint/LSTM'
    print "Check if model exist already to retrieve"
    val_batch_gen = helpers.rand_batch_gen(validX, validY, 2)
    train_batch_gen = helpers.rand_batch_gen(trainX, trainY, FLAGS.batch_size)
    ckpt = tf.train.get_checkpoint_state(ckpt_paths)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        
        sess = model.restore_last_session()
    else:
        print("Created model with fresh parameters.")
        sess = model.train(train_batch_gen, val_batch_gen)
                            
    
    
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