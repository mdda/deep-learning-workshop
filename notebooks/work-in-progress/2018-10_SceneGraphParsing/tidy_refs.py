import os, sys
import json, codecs

from collections import defaultdict

import nltk
NLTK_PATH = "nltk_data"
nltk.data.path.append(NLTK_PATH)
from nltk.corpus import wordnet

import argparse

output_path='./bist-parser/preprocess/output/'

parser = argparse.ArgumentParser()
parser.add_argument("--input",  help="Required Processed input file", type=str, default=output_path+"pre_coco_train.json.rows")
parser.add_argument("--output", help="Processed file output file",    type=str, default=output_path+"pre_coco_train_clean.json.rows")
parser.add_argument("--multiwords",  help="Create ./multiwords.json", action='store_true')
parser.add_argument("--testnodes",  help="Ensure relationship and attribute objects are present", action='store_true')
#parser.add_argument("--train",  help="Check if processed file required Training", action='store_true')
args = parser.parse_args()

refs_file = codecs.open(args.input, "r", 'utf-8')
fout      = codecs.open(args.output, 'w', encoding='utf-8')   # already in a sensible line-appending format...


multiwords_path='./multiwords.json'

if args.multiwords:
  multiwords=dict(r=defaultdict(int),a=defaultdict(int),o=defaultdict(int)) # Empty to start (relationships, attributes, objects)
else:
  if os.path.isfile(multiwords_path):
    multiwords = json.load(open(multiwords_path, 'r'))
  else:
    print("Need to create --multiwords")
    exit(0)
  

for row_num, refs_json in enumerate(refs_file):
  refs_data = json.loads(refs_json)
  refs_arr = refs_data['refs']
  
  phrases, objects, attributes, relationships  = refs_arr
  phrase=phrases[0]

  phrase=phrase.lower().replace('.', ' ').replace(',', ' ').replace('  ', ' ').strip()

  #for relationship in relationships:
  #  relatfor r in relationship:
  
  #if row_num>1000: break
  
  #print(row_num, phrase)
  
  if args.multiwords:
    for relationship in relationships:
      action=relationship[1]
      if ' ' in action.strip():
        print("Multiword action : '%s'" % (action,))
        multiwords['r'][action.strip()] += 1
        
    for attribute in attributes:
      for attr in attribute[1]:
        if ' ' in attr.strip():
          print("Multiword attribute : '%s'" % (attr,))
          multiwords['a'][attr.strip()] += 1
        
    for obj in objects:
      if ' ' in obj.strip():
        print("Multiword object : '%s'" % (obj,))
        multiwords['o'][obj.strip()] += 1

    continue

    
  obj_set = set([ obj.strip() for obj in objects ])

  if args.testnodes:
    for relationship in relationships:
      a, action, b = [ r.strip() for r in relationship]
      if a not in obj_set:
        print("Didn't find relationship object-a '%s' in '%s' for '%s'" % (a, "|".join(list(obj_set)), phrase,))
      if b not in obj_set:
        print("Didn't find relationship object-b '%s' in '%s' for '%s'" % (b, "|".join(list(obj_set)), phrase,))
      
    for attribute in attributes:
      a = attribute[0]
      if a not in obj_set:
        print("Didn't find attributed object'%s' in '%s' for '%s'" % (a, "|".join(list(obj_set)), phrase,))

    continue
    
  # First convert all the multiwords available...    
  phrase_padded = ' %s ' % phrase
  
  for relationship in relationships:
    action=relationship[1].strip()
    if ' ' in action:
      print("Multiword action : '%s' in '%s'?" % (action, phrase))
      #multiwords['r'][action.strip()] += 1
      
      if (' %s ' % action) in phrase_padded:
        print("Multiword action : '%s' found" % (action,))
      else:
        print("Multiword action : '%s' ******** NOT FOUND ********" % (action,))
        
    
    
    

if args.multiwords:    
  json.dump(multiwords, open(multiwords_path, 'w'), indent=1)
