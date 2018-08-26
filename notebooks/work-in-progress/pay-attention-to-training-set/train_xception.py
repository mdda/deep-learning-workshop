
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable

from TinyImageNet import TinyImageNet

from tensorboardX import SummaryWriter

# See https://github.com/leemengtaiwan/tiny-imagenet/blob/master/tiny-imagenet.ipynb 
# for a lot of this code
##   author: Lee Meng
##   date: 2018-08-12 12:00

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset_root = 'tiny-imagenet-200'

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].

normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64)], p=.8)

training_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    augmentation,
    transforms.ToTensor(),
    normalize])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    normalize])


in_memory = False

training_set = TinyImageNet(dataset_root, 'train', transform=training_transform, in_memory=in_memory)
valid_set    = TinyImageNet(dataset_root, 'val',   transform=valid_transform,    in_memory=in_memory)


print( training_set )

if False:
  tmpiter = iter(DataLoader(training_set, batch_size=10, shuffle=True))
  for _ in range(5):
    images, labels = tmpiter.next()
    show_images_horizontally(images, un_normalize=True)
    


#from tensorboardX import SummaryWriter
if False:
  sw = SummaryWriter(log_dir='./log', comment='xception-finetuning')
  dummy_input = Variable(torch.rand(16, 3, 64, 64)).to(device)
  sw.add_graph(resnet, (dummy_input, ))



