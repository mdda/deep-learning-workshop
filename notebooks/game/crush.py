import numpy as np


def new_board(horizontal, vertical, colours=4):
  # It's easier to work with the 'vertical flow' being along individual row vectors
  # and in this case (in the numpy array), gravity is right-to-left, and pushes up the page
  return np.random.randint(low=1, high=colours+1, size=(horizontal, vertical))
  
  
def flood_from(board, start_h, start_v, colour):
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
  

def potential_moves(b):
  # Go through the board, looking for non-zero cells to flood out of
  #   Check each cell : is anything to right-or-down of it (cannot be up or left, by construction)?
  #   If so, then this is a potential move, and should flood all adjoining cells with zero
  
  c = b.copy() # We're going to destroy this
  for h in range(0, c.shape[0]):
    for v in range(0, c.shape[1]):
      if c[h,v]>0:
        # if rightwards is a match or downwards is a match
        if (h<c.shape[0]-1 and c[h+1,h]==c[h,v]) or (v<c.shape[1]-1 and c[h,v+1]==c[h,v]):
          # Flood-fill from here
          flood_from(c, h, v, c[h,v])

if __name__ == "__main__":
  np.random.seed(1)
  b = new_board(6,17,3)
  b[0,0]=1
  b[0,1]=1
  print( b )
  
  flood_from(b, 0, 0, 1)
  print( b )
  
  
