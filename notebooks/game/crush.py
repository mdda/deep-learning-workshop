import numpy as np


def new_board(horizontal, vertical, n_colours=4):
  # It's easier to work with the 'vertical flow' being along individual row vectors
  # and in this case (in the numpy array), gravity is right-to-left, and pushes up the page
  return np.random.randint(low=1, high=n_colours+1, size=(horizontal, vertical))
  
  
def flood_from(board, start_h, start_v):  # Mutates board
  colour = board[start_h, start_v]
  # This is done non-recursively, since that probably gives us more scope for optimisation later
  stack=[ (start_h, start_v) ]
  
  while len(stack)>0:
    (h,v) = stack.pop()
    board[h,v]=0
  
    if v>0 and board[h,v-1]==colour:
      stack.append( (h,v-1) )
    if h>0 and board[h-1,v]==colour:
      stack.append( (h-1,v) )
    if v<board.shape[1]-1 and board[h,v+1]==colour:
      stack.append( (h,v+1) )
    if h<board.shape[0]-1 and board[h+1,v]==colour:
      stack.append( (h+1,v) )
  
  return board
  
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
          board[h, v]=0
        # This entry has been updated with actual cells
        filled_height += 1
    extent[h] = filled_height

  # This is the bottom-to-top gravity (TODO)
  filled_width=0
  for h in range(0, board.shape[0]):
    pass

  # if filled_width < board.shape[0], then we need to fill in new 'columns' 
  if filled_width < board.shape[0]:
    pass
    
  return board
      

def potential_moves(b):
  # Go through the board, looking for non-zero cells to flood out of
  #   Check each cell : is anything to right-or-down of it (cannot be up or left, by construction)?
  #   If so, then this is a potential move, and should flood all adjoining cells with zero
  moves=[]  # Will be filled with ( (h,v), board_after )
  
  c = b.copy() # We're going to destroy this
  for h in range(0, c.shape[0]):
    for v in range(0, c.shape[1]):
      if c[h,v]>0:
        # if rightwards is a match or downwards is a match
        if (h<c.shape[0]-1 and c[h+1,h]==c[h,v]) or (v<c.shape[1]-1 and c[h,v+1]==c[h,v]):
          # Flood-fill from here
          d = flood_from(c, h, v)
          e = apply_gravity(d.copy())
        else:
          c[h,v]=0

if __name__ == "__main__":
  np.random.seed(1)
  n_colours = 3
  b = new_board(6,17,n_colours)
  b[0,0]=1
  b[0,1]=1
  print( b )
  
  flood_from(b, 0, 0)
  print( b )
  
  apply_gravity(b, n_colours)
  
  print( b )
