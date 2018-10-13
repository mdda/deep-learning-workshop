import os, sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('orig/pytorch-openai-transformer-lm')
from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG
from text_utils import TextEncoder
from utils import ResultLogger

pretrained_model_path = os.path.join('.', 'orig', 'finetune-transformer-lm', 'model')



# So, let's read in the 

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
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    
    # Standard for pre-trained model
    #parser.add_argument('--n_embd', type=int, default=768)
    #parser.add_argument('--n_head', type=int, default=12)
    #parser.add_argument('--n_layer', type=int, default=12)
    #parser.add_argument('--embd_pdrop', type=float, default=0.1)
    #parser.add_argument('--attn_pdrop', type=float, default=0.1)
    #parser.add_argument('--resid_pdrop', type=float, default=0.1)
    #parser.add_argument('--clf_pdrop', type=float, default=0.1)
    #parser.add_argument('--afn', type=str, default='gelu')
    
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default=pretrained_model_path+'/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default=pretrained_model_path+'/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

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
    
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    
    n_special = 3
    max_len = n_ctx // 2 - 2
    
    n_ctx = min(max(
        [len(x1[:max_len]) + max(len(x2[:max_len]),
                                 len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
        + [len(x1[:max_len]) + max(len(x2[:max_len]),
                                   len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
        + [len(x1[:max_len]) + max(len(x2[:max_len]),
                                   len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
        ) + 3, n_ctx)
        
    vocab = n_vocab + n_special + n_ctx
    
    trX, trM = transform_roc(trX1, trX2, trX3)
    vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    
    #if submit:
    #    teX, teM = transform_roc(teX1, teX2, teX3)

    #n_train = len(trY)
    #n_valid = len(vaY)
    #n_batch_train = args.n_batch * max(n_gpu, 1)
    #n_updates_total = (n_train // n_batch_train) * args.n_iter

    #dh_model = DoubleHeadModel(args, clf_token, 'multiple_choice', vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)
                                                 
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

    model_tflm = TransformerModel(args)
    load_openai_pretrained_model( model_tflm, 
                                  path=pretrained_model_path+'/',
                                  path_names=os.path.join('.', 'orig', 'pytorch-openai-transformer-lm')+'/',
                                )
    model_full = model_tflm

    model_full.to(device)
    model_full = nn.DataParallel(model_full)

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
