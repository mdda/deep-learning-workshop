import numpy as np

import theano
#import theano.tensor as T
import lasagne

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

  #print(board)
  
  #print("Board mask")
  mask     = np.greater( board[:, :], 0 )*1
  feature_layers.append( mask )
  
  # This works out whether each cell is the same as the cell 'above it'
  for shift_down in [1,2,3,]:
    #print("\n'DOWN' by %d:" % (shift_down,))
    sameness = np.zeros_like(board)
    
    sameness[:,:-shift_down] = np.equal( board[:, :-shift_down], board[:, shift_down:] )*1
    #print(sameness)

    feature_layers.append( sameness )
  
  # This works out whether each cell is the same as the cell in to columns 'to the left of it'
  for shift_right in [1,2,]:
    #print("\n'RIGHT' by %d:" % (shift_right,))
    sameness = np.zeros_like(board)
    
    sameness[:-shift_right,:] = np.equal( board[:-shift_right, :], board[shift_right:, :] )*1
    #print(sameness)

    feature_layers.append( sameness )
  
  stacked = np.dstack( feature_layers )
  return np.rollaxis( stacked, 2, 0)


width, height, n_colours = 10,14,5


# Create a board for initial sizing only
board_temp = crush.new_board(width, height, n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

#print( make_features_variable_size(board_temp).shape )
print( make_features_in_layers(board_temp).shape )
#exit(0)

# Now, create a simple ?fully-connected? network (MNIST-like sizing)
#    See : https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#      Does it make sense to do dropout?  Perhaps learn over a batch a few times to 'average out' a little?
def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # No input dropout, as it tends to work less well for convolutional layers.

    # Strided and padded convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
      network, num_filters=32, filter_size=(3,3),
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(),
    )
    
    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
      network, num_filters=32, filter_size=(3,3),
      nonlinearity=lasagne.nonlinearities.rectify,
    )
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 64 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(network, p=.5),
      num_units=64,
      nonlinearity=lasagne.nonlinearities.rectify,
    )

    # And, finally, the output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(network, p=.5),
      num_units=1,
      nonlinearity=lasagne.nonlinearities.linear,  # Actually, expected score cannot be negative, but why exclude learning on one-half of plane?
    )

    return network



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


# This returns both stats for the game played and new board positions / rewards to learn from 
def play_game(game_id, model):
  training_data = []
  
  np.random.seed(game_id)
  board = crush.new_board(width, height, n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

  score_total, moves_total, game_step = 0,0,0
  while True: 
    moves = crush.potential_moves(board)
    
    moves_total += len(moves)
    
    if len(moves)==0:
      break
      
    # Choose a random move, and do it
    i = np.random.randint( len(moves) )
    
    (h,v) = moves[i]
    #print("Move : (%2d,%2d)" % (h,v))
    #crush.show_board(board, highlight=(h,v))
    
    #board, score = crush.after_move(board, h,v, -1)
    board, score = crush.after_move(board, h,v, n_colours)
    
    score_total += score
    
    print("Move : (%2d,%2d) -> Score : %3d" % (h,v,score))
    #print("  score : %d " % (score,))
    #crush.show_board(board, highlight=(0,0))

    training_data.append( make_features_in_layers(board) )
    
    game_step += 1
    
  stats=dict( steps=game_step, av_potential_moves=float(moves_total) / game_step, score=score_total )
  return stats, training_data


model=None
stats_log=[]
for i in range(0, 2):
  stats, training_data = play_game(i, model)
  
  print("steps = %d" % (stats['steps'],))
  print("average moves = %5.1f" % (stats['av_potential_moves'], ) )
  print("score_total = %d" % (stats['score'],))
  
  print( np.asarray( training_data ).shape )
  
  stats_log.append( stats )

print("DONE")

stats_cols = "steps av_potential_moves score".split()
stats_overall = np.array([ [s[c] for c in stats_cols] for s in stats_log ])

print("Min  : ",zip(stats_cols, np.min(stats_overall, axis=0).tolist()) )
print("Max  : ",zip(stats_cols, np.max(stats_overall, axis=0).tolist()) )
print("Mean : ",zip(stats_cols, np.mean(stats_overall, axis=0).tolist()) )

# Aggregate stats for 100 games (played randomly)
#('Min  : ', [('steps', 29.0), ('av_potential_moves', 7.898550724637682), ('score', 246.0)])
#('Max  : ', [('steps', 118.0), ('av_potential_moves', 16.0), ('score', 1152.0)])
#('Mean : ', [('steps', 52.68), ('av_potential_moves', 11.719583093032318), ('score', 451.58)])

