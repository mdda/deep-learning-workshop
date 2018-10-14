import os, sys

import argparse

import numpy as np
import h5py

sys.path.append('orig/pytorch-openai-transformer-lm')
from text_utils import TextEncoder

import csv

# Needed for BPE stuff
pretrained_model_path = os.path.join('.', 'orig', 'finetune-transformer-lm', 'model')

# Needed for the train/dev/test files
relation_splits_path = os.path.join('.', 'orig', 'omerlevy-bidaf_no_answer-2e9868b224e4', 'relation_splits', )


#   840000  31087632 191519344 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/train.1
#      600     21854    136415 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/dev.1
#    12000    427110   2688895 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/test.1
#   852600  31536596 194344654 total


# https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/custom-data-loader-csv.ipynb

def valid_relations(relation_file=None, only_positive=True, len_max_return=512, skip_too_long=True):
  len_max_count=0
  valid=[]
  with open(relation_file, 'r') as fp:
    reader = csv.reader(fp, delimiter='\t')
    for i, each in enumerate(reader):
      rel, ques_xxx, ques_arg, sent = each[:4]
      ques = ques_xxx.replace('XXX', ques_arg)

      if i % 10000 == 0:
        print("Line %d" % (i,))
    
      if ques_arg not in sent:
        print("MISSING ENTITY : '%s' not in '%s'" % (ques_arg, sent))
        exit(0)
    
      if only_positive and len(each)<=4:
        continue
      
      len_txt = len(ques) + len(sent) + 3
      if len_txt>len_max_return and skip_too_long:
        len_max_count+=1
        print("Skipping #%i, len_max_count=%d,pct_long=%.2f%%" % (i, len_max_count, len_max_count/i*100., ))
        continue
        
      valid.append(i)  # This is a list of the valid indices
  return valid

def save_relations(relation_phase='train', relation_fold=1, 
                   file_stub='', valid_ids=None,
                   only_positive=True, len_max_return=512, skip_too_long=True, ):
  relation_file=os.path.join( relation_splits_path, "%s.%d" % (relation_phase, relation_fold))
  file_out     =os.path.join( relation_splits_path, "%s.%d%s.hdf5" % (relation_phase, relation_fold, file_stub))
  
  if valid_ids is None:
    valid_ids = valid_relations(relation_file, only_positive=only_positive, 
                                               len_max_return=len_max_return, 
                                               skip_too_long=skip_too_long,)
  
  with h5py.File(file_out, 'w') as h5f:
    h5_data1 = h5f.create_dataset('features',
                           shape=(len(valid_ids), len_max_return),
                           compression=None,
                           dtype='int32')
    
    h5_data2 = h5f.create_dataset('labels',
                           shape=(len(valid_ids), len_max_return),
                           compression=None,
                           dtype='bool')
    
    idx=0
    with open(relation_file, 'r') as fp:
      reader = csv.reader(fp, delimiter='\t')
      for i, each in enumerate(reader):
        if i not in valid_ids: continue
        
        rel, ques_xxx, ques_arg, sent = each[:4]
        ques = ques_xxx.replace('XXX', ques_arg)
          
        indices = []
        if len(each) > 4:
          ans_list = each[4:]
          # These are offsets in characters
          indices = [(sent.index(ans), sent.index(ans) + len(ans)) for ans in ans_list]
        else:
          pass
        
        #print(rel, indices)
        #print( len(ques), len(enc_ques) )
        #if len(ques) != len(enc_ques):
        #  print( ques )
        #print( enc_ques )

        enc_ques, enc_sent = text_encoder.encode([ques, sent], verbose=False)
        print( len(ques), len(ques.split(' ')), len(enc_ques) )
        
        if i % 10000 == 0:
          print("Line %d" % (i,))
      
        if ques_arg not in sent:
          print("MISSING ENTITY : '%s' not in '%s'" % (ques_arg, sent))
          exit(0)
      
        len_txt = len(ques) + len(sent) + 3
        if len_txt>len_max_return:
          print("Truncating #%i" % (i,))
          #continue
          exit(1)
       
        #h5_data1[idx,:] = 
        
        idx+=1 
        
      
  #print(i, valid, len_max_count, len_max_count/i*100.)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_ctx', type=int, default=512)
    
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--fold',  type=int, default=1)
    
    parser.add_argument('--encoder_path', type=str, default=pretrained_model_path+'/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default=pretrained_model_path+'/vocab_40000.bpe')

    args = parser.parse_args()
    print(args)

    # Constants
    n_ctx = args.n_ctx

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    
    tokens_regular = n_vocab
    encoder['_start_']     = len(encoder)  # Last number (increments)
    encoder['_delimiter_'] = len(encoder)  # Last number (increments)
    encoder['_classify_']  = len(encoder)  # Last number (increments)
    token_clf = encoder['_classify_']
    
    tokens_special = len(encoder) - tokens_regular  # Number of extra tokens
  
    save_relations(file_stub='_pos', relation_phase='test', only_positive=True)  
    
    #yield_relations(relation_phase='train', only_positive=False)  # 832336
    #yield_relations(relation_phase='train', only_positive=True)   # 417627
    #yield_relations(relation_phase='test', only_positive=False)   #  11892
    #yield_relations(relation_phase='test', only_positive=True)    #   5940
    #exit(0)
    
    vocab = tokens_regular + tokens_special + n_ctx
    
