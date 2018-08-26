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
resize = transforms.Resize( [299, 299] )

augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64)], p=.8)

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


print( training_set )
print( valid_set    )

if False:
  tmpiter = iter(DataLoader(training_set, batch_size=10, shuffle=True))
  for _ in range(5):
    images, labels = tmpiter.next()
    show_images_horizontally(images, un_normalize=True)
   
model_base = xception.xception()  # Loads weights into model

print(model_base)



#from tensorboardX import SummaryWriter
if False:
  summary_writer = SummaryWriter(log_dir='./log', comment='xception-finetuning')
  dummy_input = torch.rand(16, 3, 64, 64).to(device)
  summary_writer.add_graph(resnet, (dummy_input, ))

exit(0)

try:
  for epoch in range(max_epochs):
    start = time.time()
    #lr_scheduler.step()
    
    epoch_loss = 0.0
    resnet.train()
    for idx, (data, target) in enumerate(trainloader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = resnet(data)
      batch_loss = ce_loss(output, target)
      batch_loss.backward()
      optimizer.step()
      epoch_loss += batch_loss.item()
  
      if idx % 10 == 0:
        print('{:.1f}% of epoch'.format(idx / float(len(trainloader)) * 100), end='\r')
      
      
    # evaluate on validation set
    num_hits = 0
    num_instances = len(valid_set)
    
    with torch.no_grad():
      resnet.eval()
      for idx, (data, target) in enumerate(validloader):
        data, target = data.to(device), target.to(device)
        output = resnet(data)
        _, pred = torch.max(output, 1) # output.topk(1) *1 = top1

        num_hits += (pred == target).sum().item()
#                 print('{:.1f}% of validation'.format(idx / float(len(validloader)) * 100), end='\r')

    valid_acc = num_hits / num_instances * 100
    print(f' Validation acc: {valid_acc}%')
    sw.add_scalar('Validation Accuracy(%)', valid_acc, epoch + 1)
        
    epoch_loss /= float(len(trainloader))
#         print("Time used in one epoch: {:.1f}".format(time.time() - start))
    
    # save model
    torch.save(resnet.state_dict(), 'models/weight.pth')
    
    # record loss
    sw.add_scalar('Running Loss', epoch_loss, epoch + 1)
        
        
except KeyboardInterrupt:
    print("Interrupted. Releasing resources...")
    
