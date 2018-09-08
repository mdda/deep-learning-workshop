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


import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  
parser.add_argument("--dataset_root", default='tiny-imagenet-200', type=str, help="directory with tiny ImageNet inside")
parser.add_argument("--checkpoint",   default=None, type=str, help="model checkpoint path to restart training")
#parser.add_argument("--epoch",        default=0, type=int, help="model checkpoint epoch")
parser.add_argument("--lr_initial",   default=0.01, type=float, help="initial lr (might be stepped down later)")
parser.add_argument("--tz",           default='Asia/Singapore', type=str, help="Timezone for local finish time estimation")

args = parser.parse_args()

dataset_root = args.dataset_root
tz = pytz.timezone(args.tz)


# See https://github.com/leemengtaiwan/tiny-imagenet/blob/master/tiny-imagenet.ipynb 
# for a lot of this code
##   author: Lee Meng
##   date: 2018-08-12 12:00

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

in_memory = False
training_set = TinyImageNet(dataset_root, 'train', transform=xception.training_transform, in_memory=in_memory)
valid_set    = TinyImageNet(dataset_root, 'val',   transform=xception.valid_transform,    in_memory=in_memory)

#print( training_set )
#print( valid_set    )

num_classes = len(training_set.label_texts)


if False:
  tmpiter = iter(DataLoader(training_set, batch_size=10, shuffle=True))
  for _ in range(5):
    images, labels = tmpiter.next()
    show_images_horizontally(images, un_normalize=True)


model_base = xception.xception_tiny_imagenet(num_classes, device)

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

if args.checkpoint is not None:
  checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  epoch_start = checkpoint['epoch']
  
  model_base.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  
  lr_scheduler = get_lr_scheduler(optimizer)
  lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
  
  print("Loaded %s - assuming epoch_now=%d" % (args.checkpoint, epoch_start,))
  
else:
  lr_scheduler = get_lr_scheduler(optimizer)



train_loader = DataLoader(training_set, batch_size=32, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_set,    batch_size=32, num_workers=4)

valid_acc_best=-1

try:
  for epoch in range(epoch_start+1, epoch_max):  # So this refers to the epoch-end value
    start = time.time()
    
    epoch_loss = 0.0
    model_base.train()
    
    for idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      
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
    
