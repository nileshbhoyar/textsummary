#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:42:43 2017

@author: nileshbhoyar
"""

import tensorflow as tf
import numpy as np
import sys
import helpers
import glob
class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001, 
            epochs=100000, model_name='seq2seq_model',attention= False,celltype = 'GRU'):
       
        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.attention = attention
        self.celltype = celltype

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # placeholders
            tf.reset_default_graph()
            sys.stdout.write('<log> Building Input variables of Graph ')
            with tf.variable_scope('decoder', reuse=True):
            #  encoder inputs : list of indices of length xseq_len
                self.enc_ip = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
                self.labels = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
                self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]
                
                
            sys.stdout.write('<log> Cells ')    
            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            with tf.variable_scope('decoder', reuse=True):
            # define the basic cell
                if self.celltype == 'GRU' :
                    basic_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(emb_dim),output_keep_prob=self.keep_prob)
                else:
                    basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                                                    output_keep_prob=self.keep_prob)
            # stack cells together : n layered model
                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


            # for parameter sharing between training model
            #  and testing model
            sys.stdout.write('<log> Building Actual model ')
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model 
                if  self.attention == True:
                    self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_attention_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                    scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input 
                #  to the next timestep
                    self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_attention_seq2seq(
                            self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                            feed_previous=True)
                else:
                    self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                    scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input 
                #  to the next timestep
                    self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                            self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                            feed_previous=True)
            

            # weighted loss
            
            sys.stdout.write('.AdamOptimizer.. ')
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # build comput graph
        __graph__()
        sys.stdout.write('</log>')



    '''
        Training and Evaluation
    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        
        batchX, batchY = train_batch_gen.next()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v
    def batch(self,BATCH_SIZE):
        #sys.stdout.write( "Training with MMAP files ")
        
        X = np.load('datasets/idx_review.npy', mmap_mode='r')
        Y =  np.load('datasets/idx_summary.npy', mmap_mode='r')
    
        i = np.random.randint(len(X))
        
        return X[i:i+BATCH_SIZE].T,Y[i:i+BATCH_SIZE].T
    def batch_eval(self,BATCH_SIZE):
        #sys.stdout.write( "Evaluating with MMAP file" )
       
        X = np.load('datasets/valid_review.npy', mmap_mode='r')
        Y =  np.load('datasets/valid_summary.npy', mmap_mode='r')
    
        i = np.random.randint(len(X))
        
        return X[i:i+BATCH_SIZE],Y[i:i+BATCH_SIZE]
      # run one batch for training
    def train_batch_from_file(self, sess, batch_size):
        # get batches
        batchX, batchY = self.batch(batch_size)
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.next()
       
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

  # evaluate 'num_batches' batches
    def eval_batches_from_file(self, sess, num_batches):
        losses = []
        trainXX,trainYY = self.batch_eval(16)
        eval_batch_gen = helpers.rand_batch_gen(trainXX, trainYY, 16)
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)
    
    def train(self, train_set, valid_set, sess=None ):
        
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
        print "training started Via All Recs"
        #sys.stdout.write('\n<log> Training started </log>\n')
        #sys.stdout.write(self.epochs)
        # run M epochs
        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i and i% (self.epochs // 1000) == 0: 
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 2) # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess
    def train_batch_file(self,batch_size,sess=None ):
        
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
        sys.stdout.write( "Training files in batch mode ")
        #sys.stdout.write('\n<log> Training started </log>\n')
        #sys.stdout.write(self.epochs)
        # run M epochs
        for i in range(self.epochs):
            try:
                self.train_batch_from_file(sess=sess,batch_size = batch_size)
                if i and i% (self.epochs // 10) == 0: 
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    self.delete_previous_checkpoints(num_previous=1)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches_from_file(sess, 2) # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess
    def __get_id(self, ckpt_file):
        """
        :param ckpt_file:
        :return:
        """
        return int(ckpt_file.split('.')[0].split('_')[1])
    def delete_previous_checkpoints(self, num_previous=1):
        """
        Deletes all previous checkpoints that are <num_previous> before the present checkpoint.
        This is done to prevent blowing out of memory due to too many checkpoints
        
        :param num_previous:
        :return:
        """
        self.present_checkpoints = glob.glob(self.ckpt_path + '/*.ckpt')
        if len(self.present_checkpoints) > num_previous:
            present_ids = [self.__get_id(ckpt) for ckpt in self.present_checkpoints]
            present_ids.sort()
            ids_2_delete = present_ids[0:len(present_ids) - num_previous]
            for ckpt_id in ids_2_delete:
                ckpt_file_nm = self.ckpt_path + self.model_name + str(ckpt_id) + '.ckpt'
                os.remove(ckpt_file_nm)
    # prediction
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
