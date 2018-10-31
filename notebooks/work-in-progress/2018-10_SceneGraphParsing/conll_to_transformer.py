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
#relation_splits_path = os.path.join('.', 'orig', 'omerlevy-bidaf_no_answer-2e9868b224e4', 'relation_splits', )


# By default, return 
def valid_relations(relation_file):
  valid, idx = [], -1
  with open(relation_file, 'r') as fp:
    reader = csv.reader(fp, delimiter='\t')
    for i, each in enumerate(reader):
      if len(each)==0:
        idx += 1 
        valid.append( idx )

      if i % 100000 == 0:
        print("Line %d, idx=%d" % (i, idx,))
      
  print("Last Line = %d" % (i,))
  return relation_file, valid

def save_relations(relation_file, valid_ids=None, file_stub='_all', bpe_max=None, save_bpe=False):
  file_out = relation_file + "%s.hdf5" % (file_stub, )
  text_out = relation_file + "%s.bpe" % (file_stub, )
  
  if bpe_max is None:
    bpe_max = n_ctx
  
  with h5py.File(file_out, 'w') as h5f:
    h5_data1 = h5f.create_dataset('features',  # These are the bpe
                           shape=(len(valid_ids), bpe_max),
                           compression=None,
                           dtype='int32')
    
    h5_data2 = h5f.create_dataset('labels',    # Types { N/a, ATTR, SUBJ, PRED, etc...)
                           shape=(len(valid_ids), bpe_max),
                           compression=None,
                           dtype='uint8')

    h5_data3 = h5f.create_dataset('deps',      # Links to next node
                           shape=(len(valid_ids), bpe_max),
                           compression=None,
                           dtype='uint8')  # >>bpe_max

    idx, bpe_truncate_count, bpe_save_arr = -1, 0, []
    """
    fout.write(str(node.id))
    fout.write("\t"+node.word)
    fout.write("\t"+(str(node.parent_id) if node.parent_id != None else '_')) 
    fout.write("\t"+(str(node.rel) if node.rel != None else '_'))
    fout.write("\t"+(str(node.prop) if node.prop != None else '_')+'\n')
    """
    bpe_maximum=0  # Track bow long we need
    
    with open(relation_file, 'r') as fp:
      reader = csv.reader(fp, delimiter='\t')
      conll_data = []
      for i, each in enumerate(reader):
        if i % 10000 == 0:
          print("Line %d" % (i,))
          
        if len(each)>0:
          conll_data.append( each )
          continue
        
        # Ok, so now conll_data has a block in the correct format...
        idx += 1
        
        if idx not in valid_ids: continue
        
        print(idx, bpe_maximum, conll_data )
        #exit(0)
        
        words, parents, relationships, properties = [], [], [], []
        for each in conll_data:
          node_id, node_word, parent_id_str, rel, prop = each
          
          word_clean = node_word.lower().replace('.', ' ').replace(',', ' ').replace('  ', ' ').strip()
          if len(word_clean)==0: 
            #continue
            word_clean='_'
          parent_id = 0 if parent_id_str=='_' else int(parent_id_str)
          
          words.append( word_clean )
          parents.append(int(parent_id))  # For OBJ which is SUBJ, this ==0, otherwise ->PRED
          relationships.append(rel)   # Maybe has 'same' indicator
          properties.append(prop)     # _/OBJ/ATTR/PRED
        
        xs_np = np.zeros((1, bpe_max), dtype=np.int32)  # bpe encoding of constructed input string
        ys_np = np.zeros((1, bpe_max), dtype=np.int8)   # class : 0=IGNORE, 1=same, 2=SUBJECT-OBJECT, 3=VERB'S-OBJECT, 4=ATTRIB
        zs_np = np.zeros((1, bpe_max), dtype=np.int8)   # position that is linked to, 0=irrelevant (a mask value)
        
        #sent_nlp  = text_encoder.nlp( sent )
        #sent_encs = text_encoder.encode_nlp(sent_nlp)

        sent_encs = text_encoder.encode_tokenized_text(words)
        sent_enc = text_encoder.flatten_bpes( sent_encs )

        # Save the bpe encoding 
        bpe_len = len(sent_enc) + 2
        
        if bpe_maximum<bpe_len:
          bpe_maximum=bpe_len  
        if bpe_len>bpe_max:
          bpe_truncate_count += 1
          print("Truncating #%i, rate = %.2f%%" % (idx, 100.*bpe_truncate_count/idx))
          trunc = bpe_max - 2
        else:
          trunc = None

        xs = [token_start] + sent_enc[:trunc] + [token_clf]
        len_xs = len(xs)
        xs_np[0, :len_xs] = xs
       
        if save_bpe:  # Append this to array to be saved to disk
          bpe_save_arr.append( text_encoder.decode( xs, inter_bpe='@@' ) )
          #print( bpe_save_arr )

        ####### HERE #######
        #exit(0)

        # Need these for answer offsets, and dependency offsets
        #sent_nlp_offsets = [ token.idx for token in sent_nlp ]
        #sent_nlp_offsets_len = len(sent_nlp_offsets)
        
        sent_enc_offsets = text_encoder.cumlen_bpes( sent_encs )

        # Go through the words, and mark the bpe locations accordingly
        for wi, word in enumerate(words):
          # ys: class : 0=IGNORE, 1=same, 2=SUBJECT-OBJECT, 3=VERB'S-OBJECT, 4=ATTRIB, 5=VERB
          cls, prop, parent = 0, properties[wi], parents[wi]
          if relationships[wi]=='same': 
            cls=1
          else:
            if prop=='OBJ':
              if parent==0: cls=2
              else: cls=3
            else:
              if prop=='ATTR': cls=4
              else:
                if prop=='PRED': cls=5
                else:
                  #print("What is this : ", word, parent, relationships[wi], prop)
                  pass  # Fall through cls=0 == ignore
        
          w_bpe = sent_enc_offsets[wi+1]  # wi+1 to account for <START>
          if w_bpe<bpe_max: # make sure we're within the bpe limit
            ys_np[0, w_bpe] = cls  # Just the first one in each word
          
            parent_bpe = sent_enc_offsets[parent] # Already is 1-based
            if parent_bpe<bpe_max:  # make sure we're pointing within the bpe limit
              zs_np[0, w_bpe] = parent_bpe

        #print(len_xs)
        #print(xs_np[0,:len_xs])
        #print(ys_np[0,:len_xs])
        #print(zs_np[0,:len_xs])
        
        if False:
          print( np.array( [xs_np[0], ys_np[0], zs_np[0]] )[:, :len_xs] )
          print()

        #print( text_encoder.decode( list( xs_np[0, :len_xs] ) ) )
        #print( list( enumerate( zip( list(xs_np[0, :len_xs]), list(ys_np[0, :len_xs]), list(zs_np[0, :len_xs])) ) ))

        #exit(0)
       
        h5_data1[idx,:] = xs_np
        h5_data2[idx,:] = ys_np
        h5_data3[idx,:] = zs_np
        
        conll_data = []  # Reset

  #print(i, valid, len_max_count, len_max_count/i*100.)
  print("Saved data to %s" % (file_out,))
  print("  bpe_maximum %d, bpe_truncate_count %d, bpe_max %d" % (bpe_maximum, bpe_truncate_count, bpe_max, ))
  
  if save_bpe:
    with open(text_out, 'w') as f:
      f.write('\n'.join(bpe_save_arr))
    print("Saved bpe data to %s" % (text_out,))
    
  return file_out

"""  Dependencies make sense!
| what party does willem drees serve ? | willem drees was born in amsterdam on july 5 , 1886 . |
[(40478, 0, 0), 

1(599, 0, 2), what 
2(2555, 0, 7), party
3(1056, 0, 7), does
4(25912, 0, 5), willem
5(975, 0, 7), (514, 0, 0), drees
7(4103, 0, 7), serve
8(257, 0, 7),  ?

(40479, 0, 0), 



10(25912, 1, 11), willem
11(975, 0, 14), (514, 0, 0), drees
13(509, 2, 14), was
14(3105, 0, 14), born
15(500, 0, 14), in 
16(23680, 0, 15), amsterdam
17(504, 0, 14),  on
18(10128, 0, 17), july
19(284, 0, 18),  5 
20(240, 0, 18), ','
21(8083, 0, 18), (35962, 0, 0),  1886
23(239, 0, 14),   '.'

(40480, 0, 0)]
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_ctx', type=int, default=128)    # Max length of input texts in bpes
    
    parser.add_argument('--path', type=str, default='./bist-parser/preprocess/output')   
    parser.add_argument('--phase', type=str, default=None)   # train, dev, None(=misc testing)
    parser.add_argument('--stub',  type=str, default='')
    parser.add_argument('--save_bpe', action='store_true')

    #parser.add_argument('--positive',  type=bool, default=False)
    
    parser.add_argument('--encoder_path', type=str, default=pretrained_model_path+'/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default=pretrained_model_path+'/vocab_40000.bpe')


    args = parser.parse_args()
    print(args)

    # Constants
    n_ctx = args.n_ctx

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    n_vocab = len(text_encoder.encoder)
    
    tokens_regular = n_vocab
    token_start = text_encoder.encoder['_start_']     = len(text_encoder.encoder)  # Last number (increments)
    token_delim = text_encoder.encoder['_delimiter_'] = len(text_encoder.encoder)  # Last number (increments)
    token_clf   = text_encoder.encoder['_classify_']  = len(text_encoder.encoder)  # Last number (increments)
    
    tokens_special = len(text_encoder.encoder) - tokens_regular  # Number of extra tokens
  
    vocab_count = tokens_regular + tokens_special

    if args.phase is not None:  # This creates the various HDF5 files - takes <5hrs for --phase=train,dev,test
      if 'train' in args.phase:  # 4h15mins ?
        train_file, valid_train_ids_all = valid_relations(args.path+'/coco_train.conll')
        train_hdf5 = save_relations(train_file, valid_ids=valid_train_ids_all, save_bpe=args.save_bpe)  # Saves ALL
        
      if 'dev' in args.phase:  # <12secs
        dev_file, valid_dev_ids_all = valid_relations(args.path+'/coco_dev.conll')
        dev_hdf5 = save_relations(dev_file, valid_ids=valid_dev_ids_all, save_bpe=args.save_bpe)  # Saves ALL
      
    
    if args.phase is None:
      s="This is a simple test of the text encoder. It's difficult to believe it will work."
      
      s_nlp = text_encoder.nlp(s)
      bpes = text_encoder.encode_nlp(s_nlp)
      print( bpes )
      
      bpe = text_encoder.flatten_bpes(bpes)
      #print( bpe )
      print( s )
      print( text_encoder.decode(bpe) )
      
      for token in s_nlp:
        # idx is a character-wise index in the original document
        print( "%3d : %2d %10s %2d %10s" % (token.idx, token.i, token.text, token.head.i, token.head.text,) )
    
    print("--token_clf=%d" % (token_clf, ))
    print("--vocab_count=%d" % (vocab_count, ))
    print("--tokens_special=%d" % (tokens_special, ))
    exit(0)
    
    
