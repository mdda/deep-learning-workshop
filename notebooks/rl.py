import numpy as np

import theano
#import theano.tensor as T

#theano.config.optimizer='fast_compile'
#theano.config.optimizer='None'

import lasagne

from game import crush


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
  mask     = np.greater( board[:, :], 0 )*1.
  feature_layers.append( mask.astype('float32') )
  
  # This works out whether each cell is the same as the cell 'above it'
  for shift_down in [1,2,3,4,5,]:
    #print("\n'DOWN' by %d:" % (shift_down,))
    sameness = np.zeros_like(board, dtype='float32')
    
    sameness[:,:-shift_down] = np.equal( board[:, :-shift_down], board[:, shift_down:] )*1.
    #print(sameness)

    feature_layers.append( sameness )
  
  # This works out whether each cell is the same as the cell in to columns 'to the left of it'
  for shift_right in [1,2,3,]:
    #print("\n'RIGHT' by %d:" % (shift_right,))
    sameness = np.zeros_like(board, dtype='float32')
    
    sameness[:-shift_right,:] = np.equal( board[:-shift_right, :], board[shift_right:, :] )*1.
    #print(sameness)

    feature_layers.append( sameness )
  
  stacked = np.dstack( feature_layers )
  return np.rollaxis( stacked, 2, 0 )


width, height, n_colours = 10,14,5


# Create a board for initial sizing only
board_temp = crush.new_board(width, height, n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

#features_shape = make_features_variable_size(board_temp).shape
features_shape = make_features_in_layers(board_temp).shape
print( features_shape )
#exit(0)

# Now, create a simple ?fully-connected? network (MNIST-like sizing)
#    See : https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#      Does it make sense to do dropout?  Perhaps learn over a batch a few times to 'average out' a little?
def build_cnn(input_var, features_shape):
    # Create a CNN of two convolution layers and a fully-connected hidden layer in front of the output layer
    
    lasagne.random.set_rng( np.random )  # np.random.RandomState.get_state()

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, features_shape[0], features_shape[1], features_shape[2]), input_var=input_var)
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
      network, num_filters=16, filter_size=(3,3),
      nonlinearity=lasagne.nonlinearities.rectify,
    )
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 64 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
      lasagne.layers.dropout(network, p=.5),
      #network,
      num_units=32,
      nonlinearity=lasagne.nonlinearities.rectify,
    )

    # And, finally, the output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
      #lasagne.layers.dropout(network, p=.5),
      network,
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
def play_game(game_id, model, per_step_discount_factor=0.95):
  training_data = dict( board=[], target=[])
  
  np.random.seed(game_id)
  board = crush.new_board(width, height, n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k

  score_total, new_cols_total, moves_total, game_step = 0,0,0,0
  while True: 
    moves = crush.potential_moves(board)
    moves_total += len(moves)
    
    if len(moves)==0:
      # Need to add a training example : This is a zero-score outcome
      training_data['board'].append( make_features_in_layers(board) )
      training_data['target'].append( 0. )
      
      break

    # Let's find the highest-scoring of those moves:  First, get all the features
    next_step_features = []
    next_step_target = []
    for (h,v) in moves:  # [0:2]
      b, score, n_cols = crush.after_move(board, h,v, -1)  # Added columns are unknown
      
      next_step_features.append( make_features_in_layers(b) )
      #next_step_target.append( score )
      next_step_target.append( n_cols )
      
    # Now evaluate the Q() values of the resulting postion for each possible move in one go
    all_features = np.array(next_step_features)  # , dtype='float32'
    #print("all_features.shape", all_features.shape)
    next_step_q = model_evaluate_features( all_features )

    next_step_aggregate = np.array( next_step_target, dtype='float32') + per_step_discount_factor * next_step_q.flatten()
    #print( next_step_aggregate )

    i = np.argmax( next_step_aggregate )
    
    ## Choose a random move, and do it
    #i = np.random.randint( len(moves) )
    
    (h,v) = moves[i]
    #print("Move : (%2d,%2d)" % (h,v))
    #crush.show_board(board, highlight=(h,v))
    
    training_data['board'].append( make_features_in_layers(board) )
    training_data['target'].append( next_step_aggregate[i] )   # This is only looking at the 'blank cols', rather than the actuals, though
    
    board, score, new_cols = crush.after_move(board, h,v, n_colours)  # Now we do the move 'for real'
    
    score_total += score
    new_cols_total += new_cols
    
    #print("Move[%2d]=(%2d,%2d) -> Score : %3d, new_cols=%1d" % (i, h,v, score,new_cols))
    #crush.show_board(board, highlight=(0,0))

    game_step += 1
    
  stats=dict( steps=game_step, av_potential_moves=float(moves_total) / game_step, score=score_total, new_cols=new_cols_total )
  return stats, training_data


board_input = theano.tensor.tensor4('inputs')
board_score = theano.tensor.vector('targets')

np.random.seed(0) # This is for the initialisation inside the CNN
model=build_cnn(board_input, features_shape)

predict_q_value  = lasagne.layers.get_output(model, deterministic=True)
estimate_q_value = lasagne.layers.get_output(model)

model_squared_error = lasagne.objectives.squared_error(estimate_q_value.reshape( (-1,) ), board_score).mean()

model_params  = lasagne.layers.get_all_params(model, trainable=True)
#model_updates = lasagne.updates.nesterov_momentum( model_squared_error, model_params, learning_rate=0.01, momentum=0.9 )

model_updates = lasagne.updates.adam( model_squared_error, model_params )
#model_updates = lasagne.updates.rmsprop( model_squared_error, model_params ) # Seems much slower to converge

model_evaluate_features = theano.function([board_input], predict_q_value)
model_train             = theano.function([board_input, board_score], model_squared_error, updates=model_updates)

def stats_aggregates(log, last=None):
  stats_cols = "steps av_potential_moves new_cols score model_err".split()
  if last:
    stats_overall = np.array([ [s[c] for c in stats_cols] for s in log[-last:] ])
  else:
    stats_overall = np.array([ [s[c] for c in stats_cols] for s in log ])

  print("Min  : ",zip(stats_cols, ["%6.1f" % (v,) for v in np.min(stats_overall, axis=0).tolist()]) )
  print("Max  : ",zip(stats_cols, ["%6.1f" % (v,) for v in np.max(stats_overall, axis=0).tolist()]) )
  print("Mean : ",zip(stats_cols, ["%6.1f" % (v,) for v in np.mean(stats_overall, axis=0).tolist()]) )
  

import datetime
t0 = datetime.datetime.now()

n_games=1*1000
batchsize=1024

stats_log=[]
training_data=dict( board=[], target=[])
for i in range(0, n_games):
  stats, training_data_new = play_game(i, model)
  
  print("game[%d]" % (i,))
  print("  steps         = %d" % (stats['steps'],))
  print("  average moves = %5.1f" % (stats['av_potential_moves'], ) )
  print("  new_cols      = %d" % (stats['new_cols'],))
  print("  score_total   = %d" % (stats['score'],))
  
  training_data['board'] += training_data_new['board']
  training_data['target'] += training_data_new['target']

  # This keeps the window from growing too big
  if len(training_data['target'])>batchsize*2:
    training_data['board'] = training_data['board'][-batchsize:]
    training_data['target'] = training_data['target'][-batchsize:]

  for iter in range(0,8):
    err = model_train( training_data['board'][-batchsize:], training_data['target'][-batchsize:] )
  
  stats['model_err'] = err
  
  stats_log.append( stats )
  
  if ((i+1) % 10)==0:
    t_now = datetime.datetime.now()
    t_elapsed = (t_now - t0).total_seconds()
    t_end_projected = t0 + datetime.timedelta( seconds=n_games* (t_elapsed/i) )
    print("    100 games in %6.1f seconds, Projected end at : %s, stored_data.length=%d" % (100.*t_elapsed/i, t_end_projected.strftime("%H:%M"), len(training_data['target']), ))
    
  if ((i+1) % 100)==0:
    stats_aggregates(stats_log, last=1000)

print("\nFINAL, overall")
stats_aggregates(stats_log)

# Aggregate stats for 100 games (played randomly)
#('Min  : ', [('steps', '  29.0'), ('av_potential_moves', '   7.9'), ('new_cols', '   0.0'), ('score', ' 246.0'), ('model_err', '  45.9')])
#('Max  : ', [('steps', ' 118.0'), ('av_potential_moves', '  16.0'), ('new_cols', '  18.0'), ('score', '1152.0'), ('model_err', '7681.9')])
#('Mean : ', [('steps', '  52.7'), ('av_potential_moves', '  11.7'), ('new_cols', '   3.0'), ('score', ' 451.6'), ('model_err', '1235.0')])

# Aggregate stats for 100 games (played with learning : ADAM per game)
#('Min  : ', [('steps', '  25.0'), ('av_potential_moves', '   6.6'), ('new_cols', '   0.0'), ('score', ' 266.0'), ('model_err', '  41.0')])
#('Max  : ', [('steps', '  98.0'), ('av_potential_moves', '  17.9'), ('new_cols', '  13.0'), ('score', ' 880.0'), ('model_err', '8107.7')])
#('Mean : ', [('steps', '  53.0'), ('av_potential_moves', '  11.1'), ('new_cols', '   3.4'), ('score', ' 451.6'), ('model_err', '1284.6')])

# Aggregate stats for 1000 games (played with learning : ADAM per game)
#('Min  : ', [('steps', '  26.0'), ('av_potential_moves', '   6.1'), ('new_cols', '   0.0'), ('score', ' 212.0'), ('model_err', '   7.8')])
#('Max  : ', [('steps', ' 139.0'), ('av_potential_moves', '  19.9'), ('new_cols', '  21.0'), ('score', '1150.0'), ('model_err', '15163.5')])
#('Mean : ', [('steps', '  54.1'), ('av_potential_moves', '  12.2'), ('new_cols', '   3.4'), ('score', ' 469.0'), ('model_err', ' 221.5')])

# Aggregate stats for 1000 games (played with learning + dropout : ADAM per game)
#('Min  : ', [('steps', '  23.0'), ('av_potential_moves', '   5.7'), ('new_cols', '   0.0'), ('score', ' 218.0'), ('model_err', '  36.2')])
#('Max  : ', [('steps', ' 180.0'), ('av_potential_moves', '  19.3'), ('new_cols', '  27.0'), ('score', '1346.0'), ('model_err', '8232.2')])
#('Mean : ', [('steps', '  53.6'), ('av_potential_moves', '  12.1'), ('new_cols', '   3.3'), ('score', ' 453.5'), ('model_err', ' 274.2')])


## AMD quad-core ('square') : 49s per 100 games (batchsize=1 game)
## i7            ('simlim') : 29s per 100 games
## gtx760 gpu    ('anson')  : 12.5s per 100 games

# Aggregate stats for 1000 games (played with learning : ADAM per game - simlim)
#('Min  : ', [('steps', '  24.0'), ('av_potential_moves', '   6.1'), ('new_cols', '   0.0'), ('score', ' 202.0'), ('model_err', '   5.6')])
#('Max  : ', [('steps', ' 161.0'), ('av_potential_moves', '  18.8'), ('new_cols', '  23.0'), ('score', '1312.0'), ('model_err', '15164.2')])
#('Mean : ', [('steps', '  53.6'), ('av_potential_moves', '  12.3'), ('new_cols', '   3.4'), ('score', ' 465.4'), ('model_err', ' 222.9')])


## AMD quad-core ('square') : 140s per 100 games (batchsize=1024 steps ~ 20 games)
## gtx760 gpu    ('anson')  :  15s per 100 games (batchsize=1024 steps ~ 20 games)
## gtx760 gpu    ('anson')  :  27s per 100 games (batchsize=1024 steps ~ 20 games, 8 iterations per training)
## Titan X gpu   ('simlim') :  27s per 100 games (batchsize=1024 steps ~ 20 games, 8 iterations per training)

# Aggregate stats for 1000 games (played with learning : ADAM per game - simlim, batchsize=1024)
#('Min  : ', [('steps', '  23.0'), ('av_potential_moves', '   7.2'), ('new_cols', '   0.0'), ('score', ' 246.0'), ('model_err', '  91.7')])
#('Max  : ', [('steps', ' 174.0'), ('av_potential_moves', '  19.4'), ('new_cols', '  33.0'), ('score', '2046.0'), ('model_err', ' 194.8')])
#('Mean : ', [('steps', '  53.8'), ('av_potential_moves', '  12.8'), ('new_cols', '   4.2'), ('score', ' 554.4'), ('model_err', ' 131.4')])

# Aggregate stats for 1000 games (played with learning : ADAM per game - simlim, batchsize=1024, 8 iterations per game)
#('Min  : ', [('steps', '  23.0'), ('av_potential_moves', '   7.4'), ('new_cols', '   0.0'), ('score', ' 232.0'), ('model_err', '  63.8')])
#('Max  : ', [('steps', ' 290.0'), ('av_potential_moves', '  18.6'), ('new_cols', '  56.0'), ('score', '3286.0'), ('model_err', ' 582.0')])
#('Mean : ', [('steps', '  69.0'), ('av_potential_moves', '  12.4'), ('new_cols', '   8.0'), ('score', ' 853.6'), ('model_err', ' 314.0')])

# simlim 1k games : last 1000:  (target = n_cols)
#('Min  : ', [('steps', '  29.0'), ('av_potential_moves', '   7.8'), ('new_cols', '   0.0'), ('score', ' 210.0'), ('model_err', '   0.0')])
#('Max  : ', [('steps', ' 217.0'), ('av_potential_moves', '  19.0'), ('new_cols', '  34.0'), ('score', '2074.0'), ('model_err', '   0.6')])
#('Mean : ', [('steps', '  71.5'), ('av_potential_moves', '  12.8'), ('new_cols', '   7.4'), ('score', ' 669.1'), ('model_err', '   0.2')])


# simlim 20k games : last 1000:
#('Min  : ', [('steps', '  26.0'), ('av_potential_moves', '   6.8'), ('new_cols', '   0.0'), ('score', ' 336.0'), ('model_err', '  15.3')])
#('Max  : ', [('steps', ' 339.0'), ('av_potential_moves', '  18.6'), ('new_cols', '  65.0'), ('score', '3572.0'), ('model_err', '1789.9')])
#('Mean : ', [('steps', '  77.7'), ('av_potential_moves', '  10.8'), ('new_cols', '  10.2'), ('score', '1070.7'), ('model_err', '  88.0')])

# simlim 100k games : last 1000:  ( interestingly, the scores peak with an average of ~1150 in 20k or so iterations)
#('Min  : ', [('steps', '  32.0'), ('av_potential_moves', '   8.4'), ('new_cols', '   0.0'), ('score', ' 242.0'), ('model_err', '1391.2')])
#('Max  : ', [('steps', ' 390.0'), ('av_potential_moves', '  20.6'), ('new_cols', '  69.0'), ('score', '2994.0'), ('model_err', '3802.2')])
#('Mean : ', [('steps', '  96.6'), ('av_potential_moves', '  13.7'), ('new_cols', '  10.7'), ('score', ' 731.2'), ('model_err', '2355.3')])


