import os

import time, pytz
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from TinyImageNet import TinyImageNet
import xception

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  
parser.add_argument("--dataset_root", default='tiny-imagenet-200', type=str, help="directory with tiny ImageNet inside")
parser.add_argument("--model",        default='xception', type=str, help="which type of model to test")
parser.add_argument("--checkpoint",   default=None, type=str, help="model checkpoint path to test", required=True)

args = parser.parse_args()

dataset_root = args.dataset_root

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

in_memory = False
score_set    = TinyImageNet(dataset_root, 'val',   transform=xception.valid_transform,    in_memory=in_memory)
num_classes = len(score_set.label_texts)

model=None
if args.model=='xception':
  model = xception.xception_tiny_imagenet(num_classes, device)
  
  
if True:
  checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  epoch_start = checkpoint['epoch']
  
  model.load_state_dict(checkpoint['model'])
  print("Loaded %s - assuming epoch_now=%d" % (args.checkpoint, epoch_start,))


score_loader = DataLoader(score_set, batch_size=16, num_workers=4)

try:
    num_hits, num_instances = 0, len(score_set)
    with torch.no_grad():
      model.eval()
      
      for idx, (data, target) in enumerate(score_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        _, pred = torch.max(output, 1) # output.topk(1) *1 = top1

        num_hits += (pred == target).sum().item()
        print('%.1f%% of scoring' % (idx / float(len(score_loader)) * 100, ), end='\r')

    score_acc = num_hits / num_instances * 100
    print("  Score acc: %.2f" % (score_acc,))

except KeyboardInterrupt:
  print("Interrupted. Releasing resources...")
    
