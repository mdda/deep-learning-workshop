import numpy as np

from game import crush

#print("Hello from rl.py")


def make_features(board):
  features = []
  #print(board)
  #print( np.pad(board, ((0,0),(1,0)), mode='constant') )
  
  #shifted_up = np.pad(board, ((0,0),(1,0)), mode='constant')[:,:-1]
  #print(shifted_up)
  #
  #sameness = np.equal( board[:,:], shifted_up )
  #print(sameness * 1)
  
  #print("Board mask")
  mask     = np.greater( board[:, :], 0 )
  #print(mask * 1)
  features.append( mask.reshape((-1,)) )
  
  # This works out whether each cell is the same as the cell 'above it'
  for shift_up in range(1,4):
    #print("\n'UP' by %d:" % (shift_up,))
    # Actually, no need for np.pad, just choose the views appropriately
    sameness = np.equal(   board[:, :-shift_up], board[:, shift_up:] )
    print(sameness * 1)
    
    mask     = np.greater( board[:, :-shift_up], 0 )
    print(mask * 1)
    
    print(np.logical_and(sameness, mask) * 1)
    
    features.append( np.logical_and(sameness, mask).reshape((-1,)) )
  
  
  #shifted_left = np.pad(board, ((1,0),(0,0)), mode='constant')[:-2,:]
  #print(shifted_left)
  
  # This works out whether each cell is the same as the cell in to columns 'to the left of it'
  for shift_left in range(1,3):
    print("\n'LEFT' by %d:" % (shift_left,))
    sameness = np.equal(   board[:-shift_left, :], board[shift_left:, :] )
    print(sameness * 1)
    
    mask     = np.greater( board[:-shift_left, :], 0 )
    print(mask * 1)
    
    print(np.logical_and(sameness, mask) * 1)
  
    features.append( np.logical_and(sameness, mask).reshape((-1,)) )
  
  #return board.reshape((-1,))
  return np.concatenate(features)


np.random.seed(1)

n_colours = 5
b = crush.new_board(10,14,n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

print( make_features(b) )
