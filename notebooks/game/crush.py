import numpy as np

#print("Hello from crush.py")

def new_board(horizontal, vertical, n_colours=5):
  # It's easier to work with the 'vertical flow' being along individual row vectors
  # and in this case (in the numpy array), gravity is right-to-left, and pushes up the page
  return np.random.randint(low=1, high=n_colours+1, size=(horizontal, vertical))
  
  
def flood_from(board, start_h, start_v):  # Mutates board
  n_cells = 0
  if board[h,v]==0: return board, n_cells
  colour = board[start_h, start_v]
  
  # This is done non-recursively, since that probably gives us more scope for optimisation later
  #   although it will repeatedly 'double back' to the cell it has just come from (because of the stack)
  stack=[ (start_h, start_v) ]
  
  while len(stack)>0:
    (h,v) = stack.pop()
    if board[h,v]!=0:
      board[h,v]=0
      n_cells += 1
  
    if v>0 and board[h,v-1]==colour:
      stack.append( (h,v-1) )
    if h>0 and board[h-1,v]==colour:
      stack.append( (h-1,v) )
    if v<board.shape[1]-1 and board[h,v+1]==colour:
      stack.append( (h,v+1) )
    if h<board.shape[0]-1 and board[h+1,v]==colour:
      stack.append( (h+1,v) )
  
  return board, n_cells
  
def apply_gravity(board, n_colours):  # Mutates board (needs n_colours to add new data on a 'column shift', if n_colours<=0 new data is -1)
  # applies left-to-right gravity, the bottom-to-top
  # it's 'stateless' in-as-much-as it doesn't even know which rows were touched in the flood before (which could save time)
  extent = np.empty( board.shape[0], dtype=int ) 
  
  for h in range(0, board.shape[0]):
    filled_height=0  # This is a mutating high-water mark for each 'column'
    # This is the right-to-left gravity (within the array)
    for v in range(0, board.shape[1]):
      if board[h,v]>0:  # if this is a filled cell
        if v>filled_height:  # And we're 'above' filled_to (i.e. to the right)
          board[h, filled_height] = board[h, v]
          #board[h, v]=0  # We'll fill this in en-mass below
        # This entry has been updated with actual cells
        filled_height += 1
    for v in range(filled_height, board.shape[1]):
      board[h, v]=0
    extent[h] = filled_height
  
  # This is the bottom-to-top gravity (TODO)
  filled_width=0
  for h in range(0, board.shape[0]):
    #print("This is the bottom-to-top gravity (TODO)")
    if extent[h]>0:   # if this is a filled column
      if h>filled_width:  # And we're 'to the left' 
        board[filled_width,:] = board[h,:] # Copy across the data
        # board[h,:] = 0 # Zero out the old version ??  No need - we'll be covering this over soon enough...
      filled_width +=1
      
  if filled_width < board.shape[0]:  # we need to fill in new 'columns' 
    #print("need to fill in new 'columns'(TODO)")
    if n_colours<=0:
      board[ filled_width:board.shape[0], : ] = -1
    else:
      board[ filled_width:board.shape[0], : ] = np.random.randint(low=1, high=n_colours+1, size=(board.shape[0]-filled_width, board.shape[1]))
    
  return board, (board.shape[0]-filled_width)
      

def potential_moves(b):
  # Go through the board, looking for non-zero cells to flood out of
  #   Check each cell : is anything to right-or-down of it (cannot be up or left, by construction)?
  #   If so, then this is a potential move, and should flood all adjoining cells with zero
  moves=[]  # Will be filled with (h,v) for each distinct scoring location/area
  
  c = b.copy() # We're going to destroy this
  for h in range(0, c.shape[0]):
    for v in range(0, c.shape[1]):
      if c[h,v]>0:
        # if rightwards is a match or downwards is a match
        if (h<c.shape[0]-1 and c[h+1,v]==c[h,v]) or (v<c.shape[1]-1 and c[h,v+1]==c[h,v]):
          # Flood-fill from here
          #   This mutates our copy, and so keeps track of 'used space'
          flood_from(c, h, v) # Don't care about return values - only mutation of c
          
          moves.append( (h,v) )
         
        #else:  # No need to blank out - this will be ignored anyway (by construction)
        #  c[h,v]=0  # This has been examined
  
  return moves


def after_move(board, h,v, n_colours):  # Returns (new board (copy), score)
  d, n_cells = flood_from(board.copy(), h, v)
  e, n_cols  = apply_gravity(d, n_colours)
  
  return e, (n_cells*(2 if n_cells<3 else (n_cells-1))), n_cols

def show_board(board, highlight=None):  # highlight=(0,0)
  d = board.copy()
  if highlight:
    d[highlight[0], highlight[1]] = -d[highlight[0], highlight[1]]  # Fixes formatting
  print(d)

  
if __name__ == "__main__":
  #np.random.seed(1)
  
  #n_colours = 3
  #b = new_board(6,17,n_colours)  # Testing
  
  n_colours = 5
  b = new_board(10,14,n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k
  
  if False:
    b[0,0]=1
    b[0,1]=1
    print( b )
    
    b, n_cells = flood_from(b, 0, 0)
    print("n_cells=%d" % (n_cells,) )
    print( b )
    
    b, n_cols  = apply_gravity(b, n_colours)
    
    print("n_cols=%d" % (n_cols,) )
    print( b )

  if False:
    moves = potential_moves(b)

    for (h,v) in moves:
      print("Move : ", (h,v))
      
      if True:
        d = b.copy()
        d[h,v] = -d[h,v]
        print(d)
      
      c, score = after_move(b, h,v, -1)
      
      print("  score : ", score)
      c[0,0]=-9  # Fixes formatting
      print(c)

  if True:
    score_total, step = 0,0
    moves_total = 0
    while True: 
      moves = potential_moves(b)
      
      moves_total += len(moves)
      
      if len(moves)==0:
        break
        
      # Choose a random move, and do it
      i = np.random.randint( len(moves) )
      
      (h,v) = moves[i]
      print("Move : (%d,%d)" % (h,v))
      if True:
        d = b.copy()
        d[h,v] = -d[h,v]
        print(d)
      
      #b, score, new_cols = after_move(b, h,v, -1)
      b, score, new_cols = after_move(b, h,v, n_colours)
      
      score_total += score
      
      print("  score : %d " % (score,))
      if True:
        d = b.copy()
        d[0,0] = -d[0,0]  # Fixes formatting
        print(d)
      
      step += 1
      
    print("steps = %d" % (step,))
    print("average moves = %5.1f" % ( float(moves_total) / step, ) )
    print("score_total = %d" % (score_total,))
