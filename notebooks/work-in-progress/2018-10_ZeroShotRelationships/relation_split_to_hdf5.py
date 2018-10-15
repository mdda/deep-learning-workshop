import os, sys

import argparse

import numpy as np
import h5py

from text_utils import TextEncoder  # This is my version
#sys.path.append('orig/pytorch-openai-transformer-lm')

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
      
      if 'Canadhan' in ques_arg:
        print("GOTCHA!")
        ques_arg = ques_arg.replace('Canadhan', 'Canadian')
      
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
                   only_positive=True, bpe_max=None, skip_too_long=False, ):
  relation_file=os.path.join( relation_splits_path, "%s.%d" % (relation_phase, relation_fold))
  file_out     =os.path.join( relation_splits_path, "%s.%d%s.hdf5" % (relation_phase, relation_fold, file_stub))
  
  if bpe_max is None:
    bpe_max = n_ctx
  
  if valid_ids is None:
    valid_ids = valid_relations(relation_file, only_positive=only_positive, 
                                               len_max_return=bpe_max*6, 
                                               skip_too_long=skip_too_long,)
  
  with h5py.File(file_out, 'w') as h5f:
    h5_data1 = h5f.create_dataset('features',
                           shape=(len(valid_ids), bpe_max),
                           compression=None,
                           dtype='int32')
    
    h5_data2 = h5f.create_dataset('labels',
                           shape=(len(valid_ids), bpe_max),
                           compression=None,
                           dtype='bool')

    def fixer(s):
      return ((' '+s+' ')
               .replace('F.C.', '#FC').replace('F.C', '#FC')
               .replace(' Jr.', ' #JUNIOR').replace(' Jr ', ' #JUNIOR ')
               .replace(' Inc.', ' #INC').replace(' Inc ', ' #INC ')
               .replace(' Bros.', ' #BROS').replace(' Bros ', ' #BROS ')
               .replace(' Co.', ' #CO').replace(' Co ', ' #CO ')
               .replace(' B.V.', ' #BV').replace(' B.V ', ' #BV ')
               .replace(' D.C.', ' #DC').replace(' D.C ', ' #DC ')
               .replace(' Mousse T. ', ' #MOUSSET ').replace(' Mousse T ', ' #MOUSSET ')
               .replace(' S.C.S.C.', ' #SCSC').replace(' S.C.S.C ', ' #SCSC ')
               .replace(' R.I.O.', ' #RIO').replace(' R.I.O ', ' #RIO ')
               .replace('S.K.', '#SK').replace('S.K', '#SK')
               .replace(' B2 K ', ' #B2K ').replace(' B2K', ' #B2K')
               .replace(' E.N.I.', ' #ENI').replace(' E.N.I ', ' #ENI ')
             ).strip()
    
    idx, bpe_truncate_count = 0, 0
    with open(relation_file, 'r') as fp:
      reader = csv.reader(fp, delimiter='\t')
      for i, each in enumerate(reader):
        if i % 10000 == 0:
          print("Line %d" % (i,))
          
        #if i<250000: continue
        #if i<410000: continue
        #if i<590000: continue
        #if i<650000: continue
        #if i<780000: continue
        if i not in valid_ids: continue
        
        rel, ques_xxx, ques_arg, sent = each[:4]
        
        if 'Canadhan' in ques_arg:
          print("GOTCHA!")
          ques_arg = ques_arg.replace('Canadhan', 'Canadian')
        
        ques = ques_xxx.replace('XXX', ques_arg)
          
        #(ques_enc, ques_clean), (sent_enc, sent_clean)
        #(ques_enc, sent_enc), (ques_clean, sent_clean) = text_encoder.encode_and_clean([ques, sent])
        
        encs, cleans, lens = text_encoder.encode_and_clean([ques, sent])
        ques_enc, sent_enc = encs
        ques_clean, sent_clean = cleans
        
        print( i, len(ques), len(ques.split(' ')), len(ques_clean.split(' ')), len(ques_enc), ques_clean )
        #print( ques ) 
        #print( ques_clean ) 
        
        bpe_ranges=[]
        if len(each) > 4:
          ans_list = each[4:]
          
          # These are offsets in characters
          #indices = [(sent.index(ans), sent.index(ans) + len(ans)) for ans in ans_list]
          
          #for ans in ans_list:
          #  s_char_start_idx = sent.index(ans) # in characters
          #  s_word_start_idx = len( sent[:s_char_start_idx-1].split(' ') )
          #  s_word_end_idx = s_word_start_idx + len( ans.split(' ') )
          #  #print( ans, (sent.split(' '))[s_word_start_idx : s_word_end_idx] )  # Seems to make sense
          # 
          #  # Now convert original sent word indices to clean word indices ...
            
          ans_encs, ans_cleans, ans_lens = text_encoder.encode_and_clean(ans_list)
          
          sent_fix = fixer(sent_clean)
          for ans_i, ans in enumerate(ans_cleans):
            ans_fix = fixer(ans)
            if ans_fix not in sent_fix:
              print("%i : ANS cleaned away! '%s' not in '%s'" % (i, ans_fix, sent_fix,) )
              exit(0)
              
            # Now we've found the ans_fix, let's figure out the bpe locations...
            s_char_start_idx = sent_fix.index(ans_fix) # in characters
            s_word_start_idx = len( sent_fix[:s_char_start_idx-1].split(' ') )
            s_word_end_idx = s_word_start_idx + len( ans_fix.split(' ') )
            
            #print( ans_fix, (sent_fix.split(' '))[s_word_start_idx : s_word_end_idx] )  # Seems to make sense = YES
            
            # So now for the bpe positions...
            # start is sum of previous bpe positions (special case for start==0)
            ans_len = ans_lens[ans_i]
            bpe_start_idx = 0
            if s_word_start_idx>0:
              bpe_start_idx=sum( ans_len[:s_word_start_idx-1] )
            bpe_end_idx  =sum( ans_len[:s_word_end_idx-1] )
            
            bpe_ranges.append( (bpe_start_idx, bpe_end_idx) )  
            
        else:
          pass
      
        if ques_arg not in sent:
          print("MISSING ENTITY : '%s' not in '%s'" % (ques_arg, sent))
          exit(0)
      
        bpe_len = len(ques_enc) + len(sent_enc) + 3
        if bpe_len>bpe_max:
          bpe_truncate_count += 1
          print("Truncating #%i, rate = %.2f%%" % (idx, 100.*bpe_truncate_count/idx))
          trunc = bpe_max - 3 - len(ques_enc) 
          
        else:
          trunc = None

        xs = [token_start] + ques_enc + [token_delim] + sent_enc[:trunc] + [token_clf]
        len_xs = len(xs)

        xs_np = np.zeros((1, bpe_max), dtype=np.int32)
        xs_np[0, :len_xs] = xs
       
        ys_np = np.zeros((1, bpe_max), dtype=np.bool)
        for bpe_start, bpe_end in bpe_ranges:
          ys_np[0, bpe_start:bpe_end] = 1
       
        h5_data1[idx,:] = xs_np
        h5_data2[idx,:] = ys_np
        
        idx+=1 
        
      
  #print(i, valid, len_max_count, len_max_count/i*100.)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_ctx', type=int, default=128)
    
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--fold',  type=int, default=1)
    parser.add_argument('--stub',  type=str, default='')
    parser.add_argument('--positive',  type=bool, default=False)
    
    parser.add_argument('--encoder_path', type=str, default=pretrained_model_path+'/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default=pretrained_model_path+'/vocab_40000.bpe')

    args = parser.parse_args()
    print(args)

    # Constants
    n_ctx = args.n_ctx

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    #encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    
    tokens_regular = n_vocab
    token_start = text_encoder.encoder['_start_']     = len(text_encoder.encoder)  # Last number (increments)
    token_delim = text_encoder.encoder['_delimiter_'] = len(text_encoder.encoder)  # Last number (increments)
    token_clf   = text_encoder.encoder['_classify_']  = len(text_encoder.encoder)  # Last number (increments)
    
    tokens_special = len(text_encoder.encoder) - tokens_regular  # Number of extra tokens
  
    if False:  # This tests the various files - takes ~2h30 for all
      save_relations(file_stub='_pos', relation_phase='train', only_positive=True)  
      save_relations(file_stub='_all', relation_phase='train', only_positive=False)  
      
      save_relations(file_stub='_pos', relation_phase='dev', only_positive=True)  
      save_relations(file_stub='_all', relation_phase='dev', only_positive=False)  
      
      #save_relations(file_stub='_pos', relation_phase='test', only_positive=True)  
      save_relations(file_stub='_all', relation_phase='test', only_positive=False)  
    
    s="This is a simple test of the text encoder. It's difficult to believe it will work."
    #encs, cleans, lens = text_encoder.encode_and_clean([s])
    #print(encs[0], cleans[0], lens[0])
    #print( text_encoder.decode(encs[0]) )

    s_nlp = text_encoder.nlp(s)
    bpes = text_encoder.encode_nlp(s_nlp)
    print( bpes )
    #bpe = [item for sublist in bpes for item in sublist]
    bpe = text_encoder.flatten_bpes(bpes)
    #print( bpe )
    print( s )
    print( text_encoder.decode(bpe) )
    
    for token in s_nlp:
      print( "%3d : %2d %10s %2d %10s" % (token.idx, token.i, token.text, token.head.i, token.head.text,) )
    
    #for token in doc:    
    exit(0)
    save_relations(file_stub=args.stub, relation_phase=args.phase, relation_fold=args.fold, only_positive=args.positive)  
    
