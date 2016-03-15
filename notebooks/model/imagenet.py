
def get_synset(path='../data/imagenet_synset_words.txt'):
  with open(path, 'r') as f:
    # Strip off the first word (until space, maxsplit=1), then synset is remainder
    return [ line.strip().split(' ', 1)[1] for line in f]

