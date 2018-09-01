import os

import time
from datetime import datetime

import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable

from TinyImageNet import TinyImageNet

from tensorboardX import SummaryWriter

import xception

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  
parser.add_argument("--dataset_root", default='tiny-imagenet-200', type=str, help="directory with tiny ImageNet inside")
parser.add_argument("--checkpoint",   default=None, type=str, help="model checkpoint path to restart training")
#parser.add_argument("--epoch",        default=0, type=int, help="model checkpoint epoch")
parser.add_argument("--lr_initial",   default=0.01, type=float, help="initial lr (might be stepped down later)")

args = parser.parse_args()


dataset_root = args.dataset_root


# See https://github.com/leemengtaiwan/tiny-imagenet/blob/master/tiny-imagenet.ipynb 
# for a lot of this code
##   author: Lee Meng
##   date: 2018-08-12 12:00

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].

# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize
# 3x299x299
resize = transforms.Resize( size=299, )   # Operates on PIL images .. interpolation=PIL.Image.BILINEAR

augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.RandomRotation(30),
    #transforms.RandomResizedCrop(64),
    transforms.RandomResizedCrop(299),
], p=.8)

normalize = transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))

training_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    resize, 
    augmentation,
    transforms.ToTensor(),
    normalize])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    resize, 
    transforms.ToTensor(),
    normalize])


in_memory = False

training_set = TinyImageNet(dataset_root, 'train', transform=training_transform, in_memory=in_memory)
valid_set    = TinyImageNet(dataset_root, 'val',   transform=valid_transform,    in_memory=in_memory)

#print( training_set )
#print( valid_set    )

num_classes = len(training_set.label_texts)


if False:
  tmpiter = iter(DataLoader(training_set, batch_size=10, shuffle=True))
  for _ in range(5):
    images, labels = tmpiter.next()
    show_images_horizontally(images, un_normalize=True)


   
model_base = xception.xception().to(device)  # Loads weights into model
#print(model_base)

# Switch off the trainability for some of the xception model 
for layer in "conv1 conv2 block1 block2 block3".split(' '):
  #model_base[layer].requires_grad = False  # Does not work...
  #print(getattr(model_base, layer))
  for p in getattr(model_base, layer).parameters():
    p.requires_grad = False

# Now substitute the last layer for what we're going to train
model_base.last_linear = torch.nn.Linear(2048, num_classes).to(device)



optimizer = torch.optim.SGD(model_base.parameters(), lr=args.lr_initial, momentum=0.9, )  # weight_decay=0.0001
#optimizer = torch.optim.Adam(model_base.parameters(), lr=args.lr_initial ) 

ce_loss = torch.nn.CrossEntropyLoss()



#from tensorboardX import SummaryWriter
if True:
  os.makedirs('./log', exist_ok=True)
  summary_writer = SummaryWriter(log_dir='./log', comment='xception-finetuning')
  
  # This should create a pretty graph.  But not yet...
  #dummy_input = torch.rand(16, 3, 64, 64).to(device)
  #summary_writer.add_graph(model_base, (dummy_input, ))


os.makedirs('./checkpoints', exist_ok=True)
epoch_start, epoch_max = 0, 50

if args.checkpoint is not None:
  checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  model_base.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  epoch_start = checkpoint['epoch']
  print("Loaded %s - assuming epoch_now=%d" % (args.checkpoint, epoch_start,))

#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.75, last_epoch=epoch_start-1) 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, )  # Measuring val_acc


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
        print('%.1f%% of epoch %d' % (idx / float(len(train_loader)) * 100, epoch-1,), end='\r')  # Python 3 FTW!
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
    print("Time used in epoch %d: %.1f" % (epoch, epoch_duration, ))
    print("  Expected finish time : %s" % ( datetime.fromtimestamp(
          (epoch_max-epoch)*epoch_duration + time.time()
        ).strftime("%A, %B %d, %Y %I:%M:%S"), ))
    print("Learning rates : ",  lr_scheduler.get_lr() )
    
    lr_scheduler.step(valid_acc)
        
    # save model
    # torch.save(model_base.state_dict(), './checkpoints/model_xception_latest.pth')

    if valid_acc_best<valid_acc:  # Save model if validation accuracy is higher than before
      torch.save(dict(
        model=model_base.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
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
    
