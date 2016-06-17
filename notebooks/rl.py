import numpy as np

from game import crush

#print("Hello from rl.py")


def make_features_variable_size(board):
  features = []
  
  #shifted_up = np.pad(board, ((0,0),(1,0)), mode='constant')[:,:-1]
  #print(shifted_up)
  
  #print("Board mask")
  mask     = np.greater( board[:, :], 0 )
  #print(mask * 1)
  features.append( mask.reshape((-1,)) )
  
  # This works out whether each cell is the same as the cell 'above it'
  for shift_up in [1,2,3,]:
    #print("\n'UP' by %d:" % (shift_up,))
    # Actually, no need for np.pad, just choose the views appropriately
    sameness = np.equal(   board[:, :-shift_up], board[:, shift_up:] )
    #print(sameness * 1)
    
    mask     = np.greater( board[:, :-shift_up], 0 )
    #print(mask * 1)
    
    #print(np.logical_and(sameness, mask) * 1)
    features.append( np.logical_and(sameness, mask).reshape((-1,)) )
  
  
  #shifted_left = np.pad(board, ((1,0),(0,0)), mode='constant')[:-2,:]
  #print(shifted_left)
  
  # This works out whether each cell is the same as the cell in to columns 'to the left of it'
  for shift_left in [1,2,]:
    #print("\n'LEFT' by %d:" % (shift_left,))
    sameness = np.equal(   board[:-shift_left, :], board[shift_left:, :] )
    #print(sameness * 1)
    
    mask     = np.greater( board[:-shift_left, :], 0 )
    #print(mask * 1)
    
    #print(np.logical_and(sameness, mask) * 1)
    features.append( np.logical_and(sameness, mask).reshape((-1,)) )
  
  #return board.reshape((-1,))
  return np.concatenate(features)


def make_features_in_layers(board):
  feature_layers = [] # These are effectively 'colours' for the CNN

  print(board)
  
  #print("Board mask")
  mask     = np.greater( board[:, :], 0 )*1
  feature_layers.append( mask )
  
  # This works out whether each cell is the same as the cell 'above it'
  for shift_down in [1,2,3,]:
    print("\n'DOWN' by %d:" % (shift_down,))
    sameness = np.zeros_like(board)
    
    # Actually, no need for np.pad, just choose the views appropriately
    sameness[:,:-shift_down] = np.equal( board[:, :-shift_down], board[:, shift_down:] )*1
    print(sameness)

    feature_layers.append( sameness )
  
  # This works out whether each cell is the same as the cell in to columns 'to the left of it'
  for shift_right in [1,2,]:
    print("\n'RIGHT' by %d:" % (shift_right,))
    sameness[:-shift_right,:] = np.equal(   board[:-shift_right, :], board[shift_right:, :] )*1
    print(sameness)

    feature_layers.append( sameness )
  
  return np.dstack( feature_layers )


np.random.seed(1)

n_colours = 5
b = crush.new_board(10,14,n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

#print( make_features_variable_size(b).shape )
print( make_features_in_layers(b).shape )

# Now, create a simple ?fully-connected? network (MNIST-like sizing)
#    See : https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#      Does it make sense to do dropout?  Perhaps learn over a batch a few times to 'average out' a little?

# If there are no actions possible Q(board)=0

# Otherwise, at random (for a given percentage tending to chose (a) more and more of the time) :

#   (a) train it to be Q(board) = R(realized,action*) + lambda*Q(realized_board, action*)
#       action* is chosen to maximise Q(state_after_action, action)
#       ( small wrinkle is that Q(next_state) may include unknown column(s) if there has been a column clearance move)

#   (b) chose action* at random (or based on some novelty measure, for instance)

# Also, if columns are added, can train Q(realized_board) = Q(state_after_action, action*)
#   so that projected boards converge to realized boards when there is a (random) extra column added


# One question is whether we learn purely on-line, or in batches
#   and if in batches, how do we store it up?  
#   Most of representation stays constant across turns (intuitively), so the data is quite 'correlated'
#   OTOH, the overall position changes much quicker than (say) chess, so perhaps it isn't too bad
#     That being said, it's not changing as quickly as the pole balancing state-space, for instance

# Perhaps just set a batchsize, and accumulate game states until it's full, then do backprop pass(es)
#   In which case, need a 'play game in a loop' function
#     accumulate stats too
#     save state every 'n' batches
#   Alternatively : Play 1 game until the end (yielding training examples as we go)

