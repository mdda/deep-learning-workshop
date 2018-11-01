import os, sys

import argparse

import numpy as np

def convert_to_conll(npz_file, bpe_file, conll_file):
  output=np.loadz(npz_file)
  f_connl=open(connl_file, 'w')
  
  idx=0
  for idx, bpe in enumerate( open(bpe_file, 'r') ):
    print(idx, bpe)
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--path',  type=str, default='./bist-parser/preprocess/output/')   
  
  parser.add_argument('--npz',   type=str, default='coco_dev.conll_v32.hdf5_v32.npz-large')   
  parser.add_argument('--bpe',   type=str, default='coco_dev.conll_v32.bpe')   
  parser.add_argument('--conll', type=str, default='coco_dev.conll_v32')   

  convert_to_conll( args.path+args.npz, args.path+args.bpe, args.path+args.conll)
    
