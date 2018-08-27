import os
import time

import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable

from TinyImageNet import TinyImageNet

from tensorboardX import SummaryWriter

import xception


# See https://github.com/leemengtaiwan/tiny-imagenet/blob/master/tiny-imagenet.ipynb 
# for a lot of this code
##   author: Lee Meng
##   date: 2018-08-12 12:00

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset_root = 'tiny-imagenet-200'

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



optimizer = torch.optim.SGD(model_base.parameters(), lr=0.01, momentum=0.9, )  # weight_decay=0.0001
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.3)

ce_loss = torch.nn.CrossEntropyLoss()



#from tensorboardX import SummaryWriter
if True:
  os.makedirs('./log', exist_ok=True)
  summary_writer = SummaryWriter(log_dir='./log', comment='xception-finetuning')
  
  # This should create a pretty graph.  But not yet...
  #dummy_input = torch.rand(16, 3, 64, 64).to(device)
  #summary_writer.add_graph(model_base, (dummy_input, ))


os.makedirs('./checkpoints', exist_ok=True)
max_epochs = 120


train_loader = DataLoader(training_set, batch_size=32, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_set,    batch_size=32, num_workers=4)

try:
  for epoch in range(max_epochs):
    start = time.time()
    lr_scheduler.step()
    
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
        print('{:.1f}% of epoch {:d}'.format(idx / float(len(train_loader)) * 100, epoch,), end='\r')
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
        print('{:.1f}% of validation'.format(idx / float(len(valid_loader)) * 100), end='\r')

    valid_acc = num_hits / num_instances * 100
    print(" Validation acc: %.2f" % (valid_acc,))
    summary_writer.add_scalar('Validation Accuracy(\%)', valid_acc, epoch + 1)
        
    epoch_loss /= float(len(train_loader))
    print("Time used in one epoch: {:.1f}".format(time.time() - start))
    
    # save model
    torch.save(model_base.state_dict(), './checkpoints/model_xception_latest.pth')
    if epoch % 10 == 0:
      torch.save(model_base.state_dict(), './checkpoints/model_xception_%04d.pth' % (epoch,))
    
    # record loss
    summary_writer.add_scalar('Running Loss', epoch_loss, epoch + 1)
        
        
except KeyboardInterrupt:
  print("Interrupted. Releasing resources...")
    
