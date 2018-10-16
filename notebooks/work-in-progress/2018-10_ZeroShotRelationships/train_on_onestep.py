import os, sys

import argparse
import random

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
    labels   = self.h5f['labels'][index]
    deps     = self.h5f['deps'][index]
    
    #if self.transform is not None:
    #  features = self.transform(features)
      
    #xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    #xmb[:, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx) # This is a single row, of batch=1
    features_with_positions = np.stack( [ features, self.postitional_encoder ], axis=1 )
    print(features.shape, features_with_positions.shape)
      
    return features_with_positions, label, deps

  def __len__(self):
    return self.num_entries


class StepwiseClassifierModel(nn.Module):
    """ Transformer with stepwise classifier(s) """
    def __init__(self, cfg, n_classifier=2, one_hot=True, vocab_count=None, n_ctx=128): # 40990
        super(StepwiseClassifierModel, self).__init__()
        self.n_embd = cfg.n_embd
        self.n_ctx = n_ctx
        self.n_classifier = n_classifier
        
        self.transformer = TransformerModel(cfg, vocab=vocab_count+n_ctx, n_ctx=n_ctx)
        self.stepwise_classifier = nn.Linear(self.n_embd, n_classifier)

        nn.init.normal_(self.stepwise_classifier.weight, std = 0.02)
        nn.init.normal_(self.stepwise_classifier.bias, 0)
        
        # Add the attention pointer idea
        self.c_attn = Conv1D(self.n_embd*2, 1, self.n_embd)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)

    def forward(self, x):   # x is the input text
        ## NO : x ~ np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)  # This is for their 0 vs 1 model
        # x ~ np.zeros((n_batch, n_ctx, 2), dtype=np.int32)     # This is more normal use-case
        # x[..., -1] is for [input_sequence, positions]
        
        h = self.transformer(x)  # These are the transformers embeddings (n_batch, n_ctx, n_embd) 
        
        #lm_logits = self.lm_head(h)
        #task_logits = self.task_head(h, x)
        #return lm_logits, task_logits

        task_logits = self.stepwise( h.view(-1, self.n_embd) ).view(-1, x.size(1), self.n_classifier)
        # Should be (n_batch, n_ctx, n_classifier)


        # Also project h on to the attention pointer
        # ~ Attention.forward
        attn = self.c_attn(h)
      
        # reshape for query and key
        query, key = attn.split(self.n_embd, dim=2)
        
        # ~ Attention.split_heads(self, x, k=False):
        new_h_shape = h.size()[:-1] + (1 , h.size(-1))  # Insert an extra dimension
        query = query.view(*new_h_shape).permute(0, 2, 1, 3)  
        key   = key.view(  *new_h_shape).permute(0, 2, 3, 1)
        
        # ~ Attention._attn(self, q, k, v):
        w = torch.matmul(query, key)
        #if True:  # self.scale:
        #  w = w / math.sqrt(self.n_embd)
        
        # Now, we have a weighting matrix (logits) over the different locations
        #w = nn.Softmax(dim=-1)(w)   # 
        print("w.size()=", w.size())
        attn_logits = w 
        
        return task_logits, attn_logits




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--desc', type=str, default='default', help="Description")
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    #parser.add_argument('--submission_dir', type=str, default='submission/')
    #parser.add_argument('--submit', action='store_true')
    #parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
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
    parser.add_argument('--bpe_path', type=str, default=pretrained_model_path+'/vocab_40000.bpe')
    
    parser.add_argument('--relation_hdf5', type=str, default='dev.1_all.hdf5')
    
    parser.add_argument('--tokens_special', type=int, default=3)  # Printed out by relation_split_to_hdf5
    parser.add_argument('--token_clf', type=int, default=40480)   # Printed out by relation_split_to_hdf5
    parser.add_argument('--vocab_count', type=int, default=40481) # Printed out by relation_split_to_hdf5
    #parser.add_argument('--n_ctx', type=int, default=128)   # Max length of input texts in bpes - get this from input hdf5 shapes

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epoch',    type=int, default=3)



    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # Constants
    #submit = args.submit
    #dataset = args.dataset
    #n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    #submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    
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

    train_loader = DataLoader(dataset=train_dataset, 
                      batch_size=args.batch_size, 
                      shuffle=False, num_workers=4)
    
    
    n_updates_total = (train_size // args.batch_size) * args.n_epoch

    model_stepwise = StepwiseClassifierModel(args, n_classifier=2, vocab_count=args.vocab_count)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(model_stepwise.parameters(),
                           lr=args.lr, schedule=args.lr_schedule, 
                           warmup=args.lr_warmup, t_total=n_updates_total,
                           b1=args.b1, b2=args.b2, e=args.e,
                           l2=args.l2, ector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
                           
    #compute_loss_fct = MultipleChoiceLossCompute(criterion,
    #                                             criterion,
    #                                             args.lm_coef,
    #                                             model_opt)
                                                 
    load_openai_pretrained_model(model_stepwise.transformer, 
                                 n_special=args.tokens_special,  n_ctx=n_ctx,   # n_ctx adjusts embedding size to include positional
                                 path=pretrained_model_path+'/',
                                 path_names=os.path.join('.', 'orig', 'pytorch-openai-transformer-lm')+'/',
                                )
    
    model_stepwise.to(device)
    model_stepwise = nn.DataParallel(model_stepwise)

    exit(0) 
    
    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        dh_model.load_state_dict(torch.load(path))
        predict(dataset, args.submission_dir)
        if args.analysis:
            rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
                                os.path.join(log_dir, 'rocstories.jsonl'))
