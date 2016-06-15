import numpy as np

from game import crush

#print("Hello from rl.py")


def make_features(board):
  return board.reshape((-1,))


np.random.seed(1)

n_colours = 5
b = crush.new_board(10,14,n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

print( make_features(b) )
