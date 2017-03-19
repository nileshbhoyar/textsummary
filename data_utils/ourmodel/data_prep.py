
"""Example of Converting TextSum model data.
Usage:
python textsum_data_convert.py --command text_to_binary --in_directories dailymail/stories --out_files dailymail-train.bin,dailymail-validation.bin,dailymail-test.bin --split 0.8,0.15,0.05
python textsum_data_convert.py --command text_to_vocabulary --in_directories cnn/stories,dailymail/stories --out_files vocab
"""

import collections
import struct
import sys
import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize

import tensorflow as tf
from tensorflow.core.example import example_pb2

from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle
from sklearn.model_selection import train_test_split
random_seed(123)  # Reproducibility

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'text_to_binary',
                           'Either text_to_vocabulary or text_to_binary.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('in_directories', '', 'path to directory')
tf.app.flags.DEFINE_string('out_files', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data')
stop_words = ['review/time' ,'review/summary','review/text','review/profileName',
              'review/score','review/userId','review/helpfulness','product/productId'
              
              ]
def _text_to_binary(input_directories, output_filenames, split_fractions):
  filenames = _get_filenames(input_directories)
  raw_data_file = filenames[1]
  print raw_data_file
  df = pd.DataFrame()
  df['Review'] = __crawl_review(raw_data_file)
  df['Summary'] =__crawl_summary(raw_data_file)
  X_train,X_test = train_test_split(df,test_size = 0.2)
  X_train_f, X_valid = train_test_split(X_train,test_size = 0.05)
  #random_shuffle(filenames)
  
  start_from_index = 0
  for index, output_filename in enumerate(output_filenames):
      if index == 0:
          print "for file %s number of records",(output_filename,len(X_train_f.index))
          _convert_files_to_binary1(X_train_f, output_filename)
      if index  == 1:
          print "for file %s number of records",(output_filename,len(X_test.index))
          _convert_files_to_binary1(X_test, output_filename)
      if index == 2:
          print "for file %s number of records",(output_filename,len(X_valid.index))
          _convert_files_to_binary1(X_valid, output_filename)
  #for index, output_filename in enumerate(output_filenames):
  #  sample_count = int(len(filenames) * split_fractions[index])
  #  print(output_filename + ': ' + str(sample_count))
  #  
  #  end_index = min(start_from_index + sample_count, len(filenames))
  #  _convert_files_to_binary(filenames[start_from_index:end_index], output_filename)
  #  
  #  start_from_index = end_index

def _text_to_vocabulary(input_directories, vocabulary_filename, max_words=200000):
  filenames = _get_filenames(input_directories)
    
  counter = collections.Counter()
    
  for filename in filenames:
    with open(filename, 'r') as f:
      document = f.read()
    
    words = document.split()
    counter.update(words)

  with open(vocabulary_filename, 'w') as writer:
    for word, count in counter.most_common(max_words - 2):
        if word not in ['review/time' ,'review/summary','review/text','review/profileName',
              'review/score','review/userId','review/helpfulness','product/productId']:
            writer.write(word + ' ' + str(count) + '\n')
    writer.write('<s> 0\n')
    writer.write('</s> 0\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')

def _get_filenames(input_directories):
  filenames = []
  for directory_name in input_directories:
    filenames.extend([join(directory_name, f) for f in listdir(directory_name) if isfile(join(directory_name, f))])
  return filenames
def _convert_files_to_binary1(pdf, output_filename):
  with open(output_filename, 'wb') as writer:
    #for filename in input_filenames:
    #    raw_data_file = filename
    #    df = pd.DataFrame()
    #    df['Review'] = __crawl_review(raw_data_file)
    #    df['Summary'] =__crawl_summary(raw_data_file)
      #with open(filename, 'r') as f:
      #  document = f.read()
        print "Number of records %d",(len(pdf.keys()))  
        for i in pdf.index:
          
          #document_parts = document.split('\n', 1)
          #assert len(document_parts) == 2
    
              title = '<d><p><s>' + pdf['Summary'][i] + '</s></p></d>'
              pdf['Review'][i]= unicode(pdf['Review'][i], errors='ignore')                        
              body = pdf['Review'][i].decode('utf8').replace('\n', ' ').replace('\t', ' ')
              sentences = sent_tokenize(body)
              body = '<d><p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p></d>'
              body = body.encode('utf8')
    
              tf_example = example_pb2.Example()
              tf_example.features.feature['article'].bytes_list.value.extend([body])
              tf_example.features.feature['abstract'].bytes_list.value.extend([title])
              tf_example_str = tf_example.SerializeToString()
              str_len = len(tf_example_str)
              writer.write(struct.pack('q', str_len))
              writer.write(struct.pack('%ds' % str_len, tf_example_str))
def _convert_files_to_binary(input_filenames, output_filename):
  with open(output_filename, 'wb') as writer:
    for filename in input_filenames:
        raw_data_file = filename
        df = pd.DataFrame()
        df['Review'] = __crawl_review(raw_data_file)
        df['Summary'] =__crawl_summary(raw_data_file)
      #with open(filename, 'r') as f:
      #  document = f.read()
        print "Number of records %d",(len(df.keys()))  
        for i in range(0,len(df.index)):
          
          #document_parts = document.split('\n', 1)
          #assert len(document_parts) == 2
    
              title = '<d><p><s>' + df['Summary'][i] + '</s></p></d>'
              df['Review'][i]= unicode(df['Review'][i], errors='ignore')                        
              body = df['Review'][i].decode('utf8').replace('\n', ' ').replace('\t', ' ')
              sentences = sent_tokenize(body)
              body = '<d><p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p></d>'
              body = body.encode('utf8')
    
              tf_example = example_pb2.Example()
              tf_example.features.feature['article'].bytes_list.value.extend([body])
              tf_example.features.feature['abstract'].bytes_list.value.extend([title])
              tf_example_str = tf_example.SerializeToString()
              str_len = len(tf_example_str)
              writer.write(struct.pack('q', str_len))
              writer.write(struct.pack('%ds' % str_len, tf_example_str))

def main(unused_argv):
  assert FLAGS.command and FLAGS.in_directories and FLAGS.out_files
  output_filenames = FLAGS.out_files.split(',')
  input_directories = FLAGS.in_directories.split(',')
  
  if FLAGS.command == 'text_to_binary':
    assert FLAGS.split
    
    split_fractions = [float(s) for s in FLAGS.split.split(',')]
    print "split fractions",split_fractions
    
    assert len(output_filenames) == len(split_fractions)
    
    _text_to_binary(input_directories, output_filenames, split_fractions)
  
  elif FLAGS.command == 'text_to_vocabulary':
    assert len(output_filenames) == 1
    
    _text_to_vocabulary(input_directories, output_filenames[0])
def __crawl_review(raw_data_file):
        """
        Crawl review
        :return: review [numpy array]
        """
        review_list = []
        print 'Crawling Reviews....'
        num_lines = 0
        with open(raw_data_file) as infile:
            for line in infile:
                if line.startswith('review/text'):
                    if num_lines >= 100000:
                        break
                    num_lines += 1
                    _,review = line.split('/text: ')
                    review_list.append(review)
        return np.array(review_list)
def __crawl_summary(raw_data_file):
        """
        Crawl summary
        :return: summary [numpy array]
        """
        summary_list = []
        print 'Crawling Summary....'
        num_lines = 0
        with open(raw_data_file) as infile:
            for line in infile:
                if line.startswith('review/summary'):
                    if num_lines >= 100000:
                        break
                    num_lines += 1
                    _,summary = line.split('/summary: ')
                    summary_list.append(summary)
        return np.array(summary_list)

if __name__ == '__main__':
  tf.app.run()




