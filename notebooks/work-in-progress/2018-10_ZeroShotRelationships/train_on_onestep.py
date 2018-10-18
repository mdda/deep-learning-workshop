import os, sys

import argparse
import random

import time, pytz
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#from text_utils import TextEncoder   # This is my version

sys.path.append('orig/pytorch-openai-transformer-lm')
from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG
from model_pytorch import Conv1D
from opt import OpenAIAdam
from utils import ResultLogger

pretrained_model_path = os.path.join('.', 'orig', 'finetune-transformer-lm', 'model')

# So, let's read in a text file
relation_splits_path = os.path.join('.', 'orig', 'omerlevy-bidaf_no_answer-2e9868b224e4', 'relation_splits', )

#   840000  31087632 191519344 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/train.1
#      600     21854    136415 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/dev.1
#    12000    427110   2688895 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/test.1
#   852600  31536596 194344654 total

# TODO : Fn to get list of relationship_types and relationship_templates for each type


# Props to : https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/custom-data-loader-csv.ipynb

class Hdf5Dataset(Dataset):
  """Custom Dataset for loading entries from HDF5 databases"""

  def __init__(self, h5_path, vocab_count, valid_indices=None): # transform=None, 
    self.h5f = h5py.File(h5_path, 'r')
    features = self.h5f['features']
    
    self.valid_indices=valid_indices          
    if valid_indices is None:
      self.num_entries = features.shape[0]
    else:
      self.num_entries = len(valid_indices)
    #self.transform = transform
    
    self.n_ctx = features.shape[1]

    self.postitional_encoder = np.arange(vocab_count, vocab_count + self.n_ctx)
      
  def __getitem__(self, index):
    if self.valid_indices is not None:  # find on-disk index
      index = self.valid_indices[index]
      
    features = self.h5f['features'][index]
    labels   = self.h5f['labels'][index].astype(np.int64)
    deps     = self.h5f['deps'][index].astype(np.int64)
    
    #if self.transform is not None:
    #  features = self.transform(features)
      
    #xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    #xmb[:, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx) # This is a single row, of batch=1

    #features_with_positions = np.stack( [ features, self.postitional_encoder ], axis=1 )
    features_with_positions = np.stack( [ features, self.postitional_encoder.copy() ], axis=1 )  # May be safer when multithreaded?
    #print(features.shape, features_with_positions.shape)  # (128,) (128, 2)

    if 3 not in list(labels):  # There is no answer to this question : Force duff values
      labels[0]=4 # end is before start
      labels[1]=3
    if 4 not in list(labels):  # There is no answer to this question : Force duff values
      labels[0]=4 # end is before start
      labels[1]=3
      
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html
    np.clip(deps, 0, self.n_ctx-1, out=deps)
    
    return features_with_positions, labels, deps

  def __len__(self):
    return self.num_entries

  def close(self):
    self.h5f.close()
    

class StepwiseClassifierModel(nn.Module):
    """ Transformer with stepwise classifier(s) """
    def __init__(self, cfg, n_classifier=None, one_hot=True, vocab_count=None, n_ctx=128): # 40990
        super(StepwiseClassifierModel, self).__init__()
        self.n_embd = cfg.n_embd
        self.n_ctx = n_ctx
        self.n_classifier = n_classifier
        
        self.transformer = TransformerModel(cfg, vocab=vocab_count+n_ctx, n_ctx=n_ctx)
        
        #self.stepwise_classifier = nn.Linear(self.n_embd, n_classifier)
        #nn.init.normal_(self.stepwise_classifier.weight, std = 0.02)
        #nn.init.normal_(self.stepwise_classifier.bias, 0)

        self.stepwise_classifier = Conv1D(n_classifier, 1, self.n_embd)
        
        # Add the attention pointer idea
        self.c_attn = Conv1D(self.n_embd*2, 1, self.n_embd)
        #self.attn_dropout = nn.Dropout(cfg.attn_pdrop)  # Not in there yet

    def forward(self, x):   # x is the input text
        ## NO : x ~ np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)  # This is for their 0 vs 1 model
        # x ~ np.zeros((n_batch, n_ctx, 2), dtype=np.int32)     # This is more normal use-case
        # x[..., -1] is for [input_sequence, positions]
        
        h = self.transformer(x)  # These are the transformers embeddings (n_batch, n_ctx, n_embd) 
        
        #lm_logits = self.lm_head(h)
        #task_logits = self.task_head(h, x)
        #return lm_logits, task_logits

        #task_logits = self.stepwise_classifier( h.view(-1, self.n_embd) )
        #.view(-1, x.size(1), self.n_classifier)
        
        task_logits = self.stepwise_classifier( h ).permute( 0, 2, 1) # CrossEntropy expects classifier to be in second position
        #print("task_logits.size()=",  task_logits.size() ) 
        #       task_logits.size()= torch.Size([8, 5, 128])  (n_batch, n_classifier, n_ctx)


        # Also project h on to the attention pointer
        # ~ Attention.forward
        attn = self.c_attn(h)
      
        # reshape for query and key
        query, key = attn.split(self.n_embd, dim=2)
        
        # ~ Attention.split_heads(self, x, k=False):
        #new_h_shape = h.size()[:-1] + (1 , h.size(-1))  # Insert an extra dimension
        #query = query.view(*new_h_shape).permute(0, 2, 1, 3)  
        #key   = key.view(  *new_h_shape).permute(0, 2, 3, 1)
        #query = query.view(*new_h_shape).permute(0, 1, 3)  
        
        # Above can be simplified, since we don't need to get too fancy...
        key   = key.permute(0, 2, 1)
        #print( "query.size()=", query.size())
        #        query.size()= torch.Size([8, 128, 768])  = batch, time_step, matcher
        #print( "key.size()=", key.size())
        #        key.size()= torch.Size([8, 768, 128])    = batch, matcher, time_step
        
        # ~ Attention._attn(self, q, k, v):
        w = torch.matmul(query, key)
        if True:  # self.scale:
          w = w / np.sqrt(self.n_embd)  # simple scaling, since we're adding up a dot product
        
        # Now, we have a weighting matrix (logits) over the different locations
        #w = nn.Softmax(dim=-1)(w)   # 
        
        #print("w.size()=", w.size())
        #       w.size()= torch.Size([8, 128, 128])  ( thinking about it : batch, time_step, position_score )

        attn_logits = w.permute( 0, 2, 1) # CrossEntropy expects classifier to be in second position ( batch, position_score, time_step )
        
        return task_logits, attn_logits


def run_predictions(test_loader=None, output_file=None):
  print("run_predictions() -> %s" % (output_file, ))
  model_stepwise.eval()

  predictions_arr, targets_arr, txts_arr = [], [], []
  for idx, (features, labels, deps) in enumerate(test_loader):
    #features, labels, deps = features.to(device), labels.to(device), deps.to(device)
    features = features.to(device)
    
    out_class_logits, out_deps_logits = model_stepwise(features)

    # Ok, so now what...
    #   Ignore the deps
    #   Just save off the out_class_logits and corresponding labels
  
    predictions_arr.append( out_class_logits.detach().cpu().numpy() )
    targets_arr.append( labels.detach().cpu().numpy() )
    
    bpes = list( features.detach().cpu().numpy()[0,:,0] )
    #txt = text_encoder.decode( bpes )
    txt = text_encoder.decode_as_fragments( bpes )
    txts_arr.append( txt )
  
    if (idx+1) % 10 == 0:
      print('%.1f%% of predictions' % (idx / float(len(test_loader)) * 100, ), end='\r')
      #break

  np.savez(output_file, predictions=np.array( predictions_arr ), targets=np.array( targets_arr ), )
  with open(output_file+'.txt', 'w') as f:
    f.write('\n'.join(txts_arr))
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint",   default=None, type=str, help="model checkpoint path to restart training")
    parser.add_argument('--stub', type=str, default='base', help="Description")
    
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    #parser.add_argument('--save_dir', type=str, default='save/')
    #parser.add_argument('--data_dir', type=str, default='data/')
    #parser.add_argument('--submission_dir', type=str, default='submission/')
    #parser.add_argument('--submit', action='store_true')
    #parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')

    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    #parser.add_argument('--n_valid', type=int, default=374)

    # Standard for pre-trained model  START
    parser.add_argument('--n_embd', type=int, default=768)  # This is the internal feature width
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--afn', type=str, default='gelu')
    # Standard for pre-trained model  END

    
    parser.add_argument('--encoder_path', type=str, default=pretrained_model_path+'/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path',     type=str, default=pretrained_model_path+'/vocab_40000.bpe')
    
    parser.add_argument('--relation_hdf5', type=str, default='dev.1_all.hdf5')
    
    parser.add_argument('--tokens_special', type=int, default=3)  # Printed out by relation_split_to_hdf5
    parser.add_argument('--token_clf',      type=int, default=40480) # Printed out by relation_split_to_hdf5
    parser.add_argument('--vocab_count',    type=int, default=40481) # Printed out by relation_split_to_hdf5
    parser.add_argument('--n_classes',      type=int, default=5)     #  #label classes = len({0, 1,2, 3,4})
    #parser.add_argument('--n_ctx', type=int, default=128)   # Max length of input texts in bpes - get this from input hdf5 shapes

    parser.add_argument('--batch_size_per_gpu', type=int, default=32)  
    parser.add_argument('--n_epoch',            type=int, default=3)
    parser.add_argument("--tz",                 type=str, default='Asia/Singapore', help="Timezone for local finish time estimation")
    
    parser.add_argument('--dep_fac',            type=float, default=0.)
    #parser.add_argument('--dep_fac',            type=float, default=0.02)
    
    parser.add_argument('--predict', action='store_true')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tz = pytz.timezone(args.tz)

    # Constants
    #log_dir = args.log_dir
    #logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(args.stub)), **args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    
    #text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    #encoder = text_encoder.encoder
    #n_vocab = len(text_encoder.encoder)
    #
    #tokens_regular = n_vocab
    #encoder['_start_']     = len(encoder)  # Last number (increments)
    #encoder['_delimiter_'] = len(encoder)  # Last number (increments)
    #encoder['_classify_']  = len(encoder)  # Last number (increments)
    #token_clf = encoder['_classify_']
    token_clf = args.token_clf
    
    #n_special = tokens_special =3  
    #tokens_special = len(encoder) - tokens_regular  # Number of extra tokens

    
    relation_hdf5 = os.path.join(relation_splits_path, args.relation_hdf5)
    
    #with h5py.File(relation_hdf5, 'r') as h5f:
    #  print(h5f['features'].shape)
    #  print(h5f['labels'].shape)
    #  print(h5f['deps'].shape)

    #n_ctx = args.n_ctx   # 
    #vocab_max = args.vocab_count + n_ctx

    train_dataset = Hdf5Dataset(h5_path=relation_hdf5, vocab_count=args.vocab_count)
    
    train_size = len(train_dataset)
    n_ctx = train_dataset.n_ctx
    
    batch_size = args.batch_size_per_gpu
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:  # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
      batch_size *= n_gpus

    n_updates_total = (train_size // batch_size) * args.n_epoch


    model_stepwise = StepwiseClassifierModel(args, n_classifier=args.n_classes, vocab_count=args.vocab_count)

    model_opt = OpenAIAdam(model_stepwise.parameters(),
                           lr=args.lr, schedule=args.lr_schedule, 
                           warmup=args.lr_warmup, t_total=n_updates_total,
                           b1=args.b1, b2=args.b2, e=args.e,
                           l2=args.l2, ector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
                           
    epoch_start, epoch_max, loss_best = -1, args.n_epoch, None

    if args.checkpoint is None:
      load_openai_pretrained_model(
        model_stepwise.transformer, 
        n_special=args.tokens_special,  n_ctx=n_ctx,   # n_ctx adjusts embedding size to include positional
        path=pretrained_model_path+'/',
        path_names=os.path.join('.', 'orig', 'pytorch-openai-transformer-lm')+'/',
      )

    model_stepwise.to(device)

    if torch.cuda.device_count() > 1:  # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model_stepwise = nn.DataParallel(model_stepwise)
      

    os.makedirs('./checkpoints', exist_ok=True)
      
    if args.checkpoint is not None:
      checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
      epoch_start = checkpoint['epoch']
      
      #from collections import OrderedDict
      #def fix_dict(state_dict):
      #  new_state_dict = OrderedDict()
      #  for k, v in state_dict.items():
      #    name = k
      #    if name.startswith('module.'):
      #      name = k[7:] # remove 'module.' of dataparallel
      #    new_state_dict[name]=v
      #  return new_state_dict
      #
      #model.load_state_dict(new_state_dict)      
      
      model_stepwise.load_state_dict(checkpoint['model'])
      model_opt.load_state_dict(checkpoint['optimizer'])
      
      #lr_scheduler = get_lr_scheduler(optimizer)
      #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      
      print("Loaded %s - assuming epoch_now=%d" % (args.checkpoint, epoch_start,))

    
    #zero = torch.zeros(1).to(device)

    if args.predict: 
      # Predict out results for all the 'relation_hdf5' instead (batch_size=1 not efficient, but 'sure')
      test_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)   # , num_workers=1
      
      from text_utils import TextEncoder   # This is my version
      text_encoder = TextEncoder(args.encoder_path, args.bpe_path)      
      
      run_predictions(test_loader=test_loader, output_file="%s_%s.npz" % (relation_hdf5, args.stub))
      exit(0)

    train_loader = DataLoader(dataset=train_dataset, 
                      batch_size=batch_size, 
                      shuffle=False)   # 2 leads to device side asserts...  , num_workers=1

    try:
      idx_loss_check, loss_recent_tot = 0, 0.
      for epoch in range(epoch_start+1, epoch_max):  # So this refers to the epoch-end value
        time_estimate_last = t_start = time.time()

        model_stepwise.train()
      
        for idx, (features, labels, deps) in enumerate(train_loader):
          features, labels, deps = features.to(device), labels.to(device), deps.to(device)
          
          model_opt.zero_grad()
          out_class_logits, out_deps_logits = model_stepwise(features)
          
          #batch_loss = ce_loss(output, target)
          
          # https://pytorch.org/docs/stable/nn.html?highlight=loss#torch.nn.BCEWithLogitsLoss
          class_loss = nn.CrossEntropyLoss(reduction='none')( out_class_logits, labels )
          #print("class_loss.size()=", class_loss.size())
          #       class_loss.size()= torch.Size([8, 128])
          class_loss_tot = class_loss.sum()
          
          # The dep loss should be ignored for those deps which == 0
          dep_loss = nn.CrossEntropyLoss(reduction='none')( out_deps_logits, deps )
          #print("dep_loss.size()=", dep_loss.size())
          #       dep_loss.size()= torch.Size([8, 128])

          #dep_loss_masked = dep_loss * ( deps>0 )  # This zeros out all positions where deps == 0
          #dep_loss_masked = torch.where(deps>0, dep_loss, zero)  # This zeros out all positions where deps == 0
          #dep_loss_tot = dep_loss_masked.sum() / batch_size
          dep_loss_tot = dep_loss.masked_fill_( deps==0, 0. ).sum()
          
          factor_hints="Factor hints (class_loss=%8.4f, deps_loss=%10.4f, fac=%.8f)" % ( class_loss_tot.item(), dep_loss_tot.item(), class_loss_tot.item()/dep_loss_tot.item(), )
          #factor hints :  (231.14927673339844, 225.23297119140625, 1.0262674932124587)

          batch_loss = class_loss_tot + args.dep_fac * dep_loss_tot
          
          batch_loss.backward()
          
          model_opt.step()
          
          loss_this = batch_loss.item()
          loss_recent_tot += loss_this
          
          if idx % 10 == 0:
            print('%.1f%% of epoch %d' % (idx / float(len(train_loader)) * 100, epoch,), end='\r')  # Python 3 FTW!

          if idx % 100 == 0:
            print(factor_hints)

          sentences_since_last_check = (idx-idx_loss_check)*batch_size
          #if sentences_since_last_check > 50000:  # Potentially save every  50000 sentences  (~30mins on TitanX)
          if sentences_since_last_check > 200000:  # Potentially save every 200000 sentences  (~2hrs on TitanX)
            loss_recent = loss_recent_tot / float(sentences_since_last_check)   # loss per sentence
          
            if loss_best is None or loss_recent<loss_best:  # Save model if loss has decreased
              fname = './checkpoints/model-stepwise_%s_%04d-%06d.pth' % (args.stub, epoch, idx*batch_size,)
              print("Saving Checkpoint : '%s', loss_recent=%.4f" % (fname, loss_recent, ))
              torch.save(dict(
                epoch=epoch,
                model=model_stepwise.state_dict(), 
                optimizer=model_opt.state_dict(), 
                #lr_scheduler=lr_scheduler.state_dict(), 
              ), fname)
              loss_best=loss_recent
              idx_loss_check, loss_recent_tot = idx, 0.  # Restart running tallies
          
          t_now = time.time()
          if t_now - time_estimate_last>5*60.: # Update every 5 minutes
            calc_duration = t_now-t_start
            calc_fraction = (idx*batch_size)/len(train_dataset)
            epoch_duration = calc_duration/calc_fraction
            epoch_max_secs = (epoch_max-(epoch+calc_fraction))*epoch_duration 
            epoch_max_end = epoch_max_secs + time.time()  # This is since the epoch in seconds
            print("Time used for %.2f of epoch %d: %.1f seconds" % (calc_fraction, epoch, calc_duration, ))
            print("  Time per 1000 lines : %.3f seconds" % (epoch_duration/len(train_dataset)*1000., ))
            print("  Expected finish in : %.2f hours" % ( epoch_max_secs/60/60, ))
            print("  Expected finish time : %s (server)"  % ( datetime.fromtimestamp(epoch_max_end).strftime("%A, %B %d, %Y %H:%M:%S %Z%z"), ))
            print("  Expected finish time : %s (local)"   % ( datetime.fromtimestamp(epoch_max_end, timezone.utc).astimezone(tz=tz).strftime("%A, %B %d, %Y %H:%M:%S %Z%z"), ))
            
            #print("  Expected finish time : %s (local)"  % ( tz.localize(datetime.fromtimestamp(epoch_max_end)).strftime("%A, %B %d, %Y %H:%M:%S %Z%z"), ))
            
            time_estimate_last = time.time()  # Keep track of estimate times
        
            # https://stackoverflow.com/questions/1398674/display-the-time-in-a-different-time-zone
        
            #  import time, pytz
            #  from datetime import datetime, timezone
            #  datetime.fromtimestamp(1247634536).strftime("%A, %B %d, %Y %H:%M:%S %Z%z")
            #  datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %H:%M:%S %Z%z")  # GCP appears to report in UTC
            #  datetime.fromtimestamp(time.time(), timezone.utc).strftime("%A, %B %d, %Y %I:%M:%S %Z%z") 
            #  tz = pytz.timezone('Asia/Singapore')
            #  datetime.fromtimestamp(time.time(), timezone.utc).astimezone(tz=tz).strftime("%A, %B %d, %Y %H:%M:%S %Z%z")

        
        idx_loss_check -= len(train_dataset)/batch_size  # Keep track of reset idxs
        
        #epoch_duration = time.time()-start
        #epoch_max_end = (epoch_max-epoch)*epoch_duration + time.time()
        #print("Time used in epoch %d: %.1f seconds" % (epoch, epoch_duration, ))
        #print("  Expected finish time : %s (server)" % ( datetime.fromtimestamp(epoch_max_end).strftime("%A, %B %d, %Y %H:%M:%S %Z%z"), ))
        #print("  Expected finish time : %s (local)"  % ( datetime.fromtimestamp(epoch_max_end).astimezone(tz).strftime("%A, %B %d, %Y %H:%M:%S %Z%z"), ))

        # End-of-epoch saving
        fname = './checkpoints/model-stepwise_%s_%04d-%06d_end-epoch.pth' % (args.stub, epoch, idx*batch_size,)
        print("Saving End-epoch checkpoint : '%s'" % (fname, ))
        torch.save(dict(
          epoch=epoch,
          model=model_stepwise.state_dict(), 
          optimizer=model_opt.state_dict(), 
        ), fname)
        
    except KeyboardInterrupt:
      print("Interrupted. Releasing resources...")
    
    finally:
      train_dataset.close()

    exit(0) 
