import os

import time, pytz
from datetime import datetime

import torch

#import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable

from TinyImageNet import TinyImageNet
import xception

from tensorboardX import SummaryWriter
#from collections import Counter  # For the histogram debug printing

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  
parser.add_argument("--dataset_root", default='tiny-imagenet-200', type=str, help="directory with tiny ImageNet inside")
parser.add_argument("--checkpoint",   default=None, type=str, help="xception checkpoint path for training example featurisation")
parser.add_argument("--trainvalues",  default='tiny-imagenet-200_trainval.pth', type=str, help="file containing values+labels for training set")
parser.add_argument("--checkpoint_judge",   default=None, type=str, help="judge model checkpoint path to restart training")
parser.add_argument("--lr_initial",   default=0.01, type=float, help="initial lr (might be stepped down later)")
parser.add_argument("--tz",           default='Asia/Singapore', type=str, help="Timezone for local finish time estimation")

args = parser.parse_args()

dataset_root = args.dataset_root
tz = pytz.timezone(args.tz)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

in_memory = False
training_set = TinyImageNet(dataset_root, 'train', transform=xception.training_transform, in_memory=in_memory)
#training_set = TinyImageNet(dataset_root, 'train', transform=xception.valid_transform, in_memory=in_memory)
valid_set    = TinyImageNet(dataset_root, 'val',   transform=xception.valid_transform,    in_memory=in_memory)

num_classes = len(training_set.label_texts)


if True:  # Create the featuriser
  model_base = xception.xception_tiny_imagenet(num_classes, device)
  
  checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  model_base.load_state_dict(checkpoint['model'])

  model_base = xception.make_headless(model_base, device)
  model_base.eval()


if True:  # Load the trainingset features and labels
  #torch.save(dict(targets=targets, features=features,), args.save_trainvalues)
  checkpoint_trainingset = torch.load(args.trainvalues)
  features_trainingset = torch.tensor( checkpoint_trainingset['features'], device=device, dtype=torch.float32, requires_grad=False)
  targets_trainingset  = torch.tensor( checkpoint_trainingset['targets'],  device=device, dtype=torch.long, requires_grad=False)

  features_means = torch.mean( features_trainingset, dim=0, keepdim=True )
  features_std   = torch.std(  features_trainingset, dim=0, keepdim=True )  # impicitly removes mean
  
  # Per-location normalization (this makes each feature column equally 'valid')
  features_trainingset = (features_trainingset-features_means)/features_std
  
  #print( torch.mean( features_trainingset, dim=0 ).cpu()[0:10] )
  #print( torch.std(  features_trainingset, dim=0 ).cpu()[0:10] )
  
  # Now normalise each feature vector's norm itself
  features_trainingset = features_trainingset / torch.norm( features_trainingset, p=2., dim=1, keepdim=True, )
  #print( torch.norm( features_trainingset, p=2., dim=1 ).cpu()[0:10] )

# Here, let's build the 'judge model' == 'model'
#  SOON





optimizer = torch.optim.SGD(model_base.parameters(), lr=args.lr_initial, momentum=0.9, )  # weight_decay=0.0001
#optimizer = torch.optim.Adam(model_base.parameters(), lr=args.lr_initial ) 

ce_loss = torch.nn.CrossEntropyLoss()



if True:
  os.makedirs('./log', exist_ok=True)
  summary_writer = SummaryWriter(log_dir='./log', comment='xception-finetuning')
  
  # This should create a pretty graph.  But not yet...
  #dummy_input = torch.rand(16, 3, 64, 64).to(device)
  #summary_writer.add_graph(model_base, (dummy_input, ))


os.makedirs('./checkpoints', exist_ok=True)
epoch_start, epoch_max = 0, 50

def get_lr_scheduler(opt):
  #return torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.75, last_epoch=epoch_start-1) 
  return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, verbose=True, )  # Measuring val_acc

if args.checkpoint_judge is not None:
  checkpoint = torch.load(args.checkpoint_judge, map_location=lambda storage, loc: storage)
  epoch_start = checkpoint['epoch']
  
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  
  lr_scheduler = get_lr_scheduler(optimizer)
  lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
  
  print("Loaded %s - assuming epoch_now=%d" % (args.checkpoint_judge, epoch_start,))
  
else:
  lr_scheduler = get_lr_scheduler(optimizer)


train_loader = DataLoader(training_set, batch_size=32, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_set,    batch_size=32, num_workers=4)


valid_acc_best=-1

try:
  for epoch in range(epoch_start+1, epoch_max):  # So this refers to the epoch-end value
    start = time.time()
    
    epoch_loss = 0.0
    #model.train()
    
    for idx, (data, target) in enumerate(train_loader):
    #for idx, (data, target) in enumerate(valid_loader):
      data, target = data.to(device), target.to(device)
      
      # Let's just go through the batch, and find the matching training set examples
      #   based on their dot product (?) with the features from each training example arriving
      
      features = model_base(data)
      #print(data.size(), features.size(), ) # torch.Size([32, 3, 299, 299]) torch.Size([32, 2048])
      
      # Normalise column-wise the features 
      features_norm = (features-features_means)/features_std      
      
      # And also make them norm2 = 1
      features_norm = features_norm / torch.norm(features_norm, p=2., dim=1, keepdim=True) 
      
      # Get the Bx(training_set_size) 'attention factors'
      attention = torch.matmul( features_norm, features_trainingset.transpose(0,1) )

      attn_weights, attn_idx = torch.sort(attention, dim=1, descending=True, )  # Highest match at position [b, 0]
      
      for b in range( data.size(0) ):  # Each member of the batch
        top_n = [ targets_trainingset[ attn_idx[b,i] ].cpu().item() for i in range(16) ]
        top_n = top_n[1:]  # Take off the first entry
        
        counts = sorted( [ (top_n.count(x), x) for x in set(top_n)], reverse=True )  # Not super-fast
        #print(counts)
        
        print("Target=%3d, Found : %s, Weights: %s" % (
          target[b].cpu(), 
          #', '.join([ ("%+5.2f->%3d" % (attn_weights[b,i].cpu(), targets_trainingset[ attn_idx[b,i] ].cpu(), )) for i in range(16)]),
          #', '.join([ ("%3d" % (targets_trainingset[ attn_idx[b,i] ].cpu(), )) for i in range(1, 16)]),  # Skip first one
          ', '.join([ ("%3d" % (targets_trainingset[ attn_idx[b,i] ].cpu(), )) for i in range(16)]),  # Use all
          #', '.join([ ("%+5.2f" % (attn_weights[b,i].cpu(), )) for i in range(16)]),
          ', '.join([ ("%2d->%3d" % (c, i)) for c, i in counts ]),
          ))
      
      exit(0)
      
      
      
      optimizer.zero_grad()
      output = model_base(data)
      
      batch_loss = ce_loss(output, target)
      batch_loss.backward()
      
      optimizer.step()
      epoch_loss += batch_loss.item()
  
      if idx % 10 == 0:
        print('%.1f%% of epoch %d' % (idx / float(len(train_loader)) * 100, epoch,), end='\r')  # Python 3 FTW!
        #break
      
    # evaluate on validation set
    num_hits, num_instances = 0, len(valid_set)
    with torch.no_grad():
      model_base.eval()
      
      for idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)
        output = model_base(data)
        
        _, pred = torch.max(output, 1) # output.topk(1) *1 = top1

        num_hits += (pred == target).sum().item()
        print('%.1f%% of validation' % (idx / float(len(valid_loader)) * 100, ), end='\r')

    valid_acc = num_hits / num_instances * 100
    print(" Epoch %d validation acc: %.2f" % (epoch, valid_acc,))
    summary_writer.add_scalar('Validation Accuracy(\%)', valid_acc, epoch)

    epoch_loss /= float(len(train_loader))
    epoch_duration = time.time()-start
    epoch_max_end = (epoch_max-epoch)*epoch_duration + time.time()
    print("Time used in epoch %d: %.1f" % (epoch, epoch_duration, ))
    print("  Expected finish time : %s (server)" % ( datetime.fromtimestamp(epoch_max_end).strftime("%A, %B %d, %Y %I:%M:%S %Z%z"), ))
    print("  Expected finish time : %s (local)"  % ( datetime.fromtimestamp(epoch_max_end).astimezone(tz).strftime("%A, %B %d, %Y %I:%M:%S %Z%z"), ))
        
    if False: # For Step (not ReduceOnPlateau)
      print("Learning rates : ",  lr_scheduler.get_lr() )
    
    if True:  # for ReduceOnPlateau
      lr_scheduler.step(valid_acc)
        
    # save model
    # torch.save(model_base.state_dict(), './checkpoints/model_xception_latest.pth')

    if valid_acc_best<valid_acc:  # Save model if validation accuracy is higher than before
      torch.save(dict(
        epoch=epoch,
        model=model_base.state_dict(), 
        optimizer=optimizer.state_dict(), 
        lr_scheduler=lr_scheduler.state_dict(), 
      ), './checkpoints/model_xception_%04d.pth' % (epoch,))
      valid_acc_best=valid_acc
    
    # save the model with a rolling window
    #checkpoint_old = './checkpoints/model_xception_%04d.pth' % (epoch-3,)
    #if True and os.path.isfile(checkpoint_old):
    #  os.remove(checkpoint_old)
    
    # record loss
    summary_writer.add_scalar('Running Loss', epoch_loss, epoch)
        
        
except KeyboardInterrupt:
  print("Interrupted. Releasing resources...")
    
