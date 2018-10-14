import os, sys

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('orig/pytorch-openai-transformer-lm')
from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import ResultLogger

import csv

pretrained_model_path = os.path.join('.', 'orig', 'finetune-transformer-lm', 'model')

# So, let's read in a text file
relation_splits_path = os.path.join('.', 'orig', 'omerlevy-bidaf_no_answer-2e9868b224e4', 'relation_splits', )


#   840000  31087632 191519344 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/train.1
#      600     21854    136415 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/dev.1
#    12000    427110   2688895 orig/omerlevy-bidaf_no_answer-2e9868b224e4/relation_splits/test.1
#   852600  31536596 194344654 total

def yield_relations(relation_phase='train', relation_split=1, only_positive=True, len_max_return=512):
  relation_file=os.path.join( relation_splits_path, "%s.%d" % (relation_phase, relation_split))
  valid, len_max_txt, len_max_count = 0,0,0
  with open(relation_file, 'r') as fp:
    reader = csv.reader(fp, delimiter='\t')
    for i, each in enumerate(reader):
      rel, ques_xxx, ques_arg, sent = each[:4]
      ques = ques_xxx.replace('XXX', ques_arg)
        
      indices = []
      if len(each) > 4:
        ans_list = each[4:]
        # These are offsets in characters
        indices = [(sent.index(ans), sent.index(ans) + len(ans)) for ans in ans_list]
        #starts, ends = zip(*indices)
        #ans_start = min(starts)
        #ans_end = max(ends)
        #ans = sent[ans_start:ans_end]
        #qa['answers'].append({'text': ans, 'answer_start': ans_start})
      else:
        # Negative example
        pass
            
      #print(rel, indices)
      #enc_ques = text_encoder.encode(ques, verbose=False)
      #print( len(ques), len(enc_ques) )
      #if len(ques) != len(enc_ques):
      #  print( ques )
      #print( enc_ques )
    
      if i % 10000 == 0:
        print("Line %d" % (i,))
    
      if ques_arg not in sent:
        print("MISSING ENTITY : '%s' not in '%s'" % (ques_arg, sent))
    
      if only_positive and len(indices)==0:
        continue
      
      len_txt = len(ques) + len(sent) + 3
      
      if len_max_txt<len_txt:
        len_max_txt=len_txt
        #print( ques, sent )
        #print(len_max_txt)
        
      if len_txt>len_max_return:
        len_max_count+=1
        print("Skipping #%i, len_max_count=%d, len_max_txt=%d, pct_long=%.2f%%" % (i, len_max_count, len_max_txt,  len_max_count/i*100., ))
        continue
        
      valid += 1  # We're going to want this for the hdf5 accessible version
      
      #if i>1000: break
      
      #yield i, ques, sent, indices
      
  print(i, valid, len_max_count, len_max_txt,  len_max_count/i*100.)

# TODO : Fn to get list of relationship_types and relationship_templates for each type



class StepwiseClassifierModel(nn.Module):
    """ Transformer with stepwise classifier(s) """
    def __init__(self, cfg, n_classifier=2, one_hot=True, vocab=40990, n_ctx=512):
        super(StepwiseClassifierModel, self).__init__()
        self.n_embd = cfg.n_embd
        self.n_ctx = cfg.n_ctx
        self.n_classifier = n_classifier
        
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.stepwise = nn.Linear(self.n_embd, n_classifier)

        nn.init.normal_(self.stepwise.weight, std = 0.02)
        nn.init.normal_(self.stepwise.bias, 0)

    def forward(self, x):   # x is the input text
        # x ~ np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)  # This is for their 0 vs 1 model
        # x ~ np.zeros((n_batch, n_ctx, 2), dtype=np.int32)     # This is more normal use-case
        # x[..., -1] is for [input_sequence, positions]
        
        h = self.transformer(x)  # These are the transformers embeddings (n_batch, n_ctx, n_embd) 
        
        #lm_logits = self.lm_head(h)
        #task_logits = self.task_head(h, x)
        #return lm_logits, task_logits

        task_logits = self.stepwise( h.view(-1, self.n_embd) ).view(-1, x.size(1), self.n_classifier)
        # Should be (n_batch, n_ctx, n_classifier)
        
        return task_logits




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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    
    # Standard for pre-trained model  START
    parser.add_argument('--n_embd', type=int, default=768)
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

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # Constants
    #submit = args.submit
    #dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    #submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    #print("Encoding dataset...")
    #((trX1, trX2, trX3, trY),
    # (vaX1, vaX2, vaX3, vaY),
    # (teX1, teX2, teX3)) = encode_dataset(*rocstories(data_dir, n_valid=args.n_valid),
    #                                      encoder=text_encoder)
    
    tokens_regular = n_vocab
    encoder['_start_']     = len(encoder)  # Last number (increments)
    encoder['_delimiter_'] = len(encoder)  # Last number (increments)
    encoder['_classify_']  = len(encoder)  # Last number (increments)
    token_clf = encoder['_classify_']
    
    #n_special = tokens_special =3  
    tokens_special = len(encoder) - tokens_regular  # Number of extra tokens
    
    #yield_relations(relation_phase='train', only_positive=False)  # 832336
    #yield_relations(relation_phase='train', only_positive=True)   # 417627
    #yield_relations(relation_phase='test', only_positive=False)   #  11892
    #yield_relations(relation_phase='test', only_positive=True)    #   5940
    exit(0)
    
    #max_len = n_ctx // 2 - 2
    
    #n_ctx = min(max(
    #    [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                             len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
    #    + [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                               len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
    #    + [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                               len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
    #    ) + 3, n_ctx)
    
    #n_ctx = 100 # max length of encoded strings...
        
    vocab = tokens_regular + tokens_special + n_ctx
    
    #trX, trM = transform_roc(trX1, trX2, trX3)
    #vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    
    #if submit:
    #    teX, teM = transform_roc(teX1, teX2, teX3)

    #n_train = len(trY)
    #n_valid = len(vaY)
    #n_batch_train = args.n_batch * max(n_gpu, 1)
    #n_updates_total = (n_train // n_batch_train) * args.n_iter
    
    n_updates_total = (840000 // args.batch_size) * args.n_epoch

    model_stepwise = StepwiseClassifierModel(args, n_classifier=2, vocab=vocab, n_ctx=n_ctx)

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
                                                 
    load_openai_pretrained_model(model_stepwise.transformer, n_ctx=n_ctx, n_special=tokens_special,
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
