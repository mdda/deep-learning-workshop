
## Helpful stuff :

# https://jakevdp.github.io/blog/2013/06/01/ipython-notebook-javascript-python-communication/
# https://github.com/fluxtream/fluxtream-ipy/
#   blob/master/Communication%20between%20kernel%20and%20javascript%20in%20iPython%202.0.ipynb

def display_via_javascript_script(element_id, board):
  #print("display_via_javascript_script")
  a = [ "[%s]" % (','.join([ "%d" % (c,) for c in board[h].tolist() ]),) for h in range(0, board.shape[0]) ]
  return("""<script type="text/javascript">display_board("%s",[%s])</script>""" % (element_id, ','.join(a),))
  #return("<b>HelloWorld</b>")

def render_to_json(board):
  return([ [ c for c in board[h].tolist() ] for h in range(0,board.shape[0]) ])
  
  #a = [ "[%s]" % (','.join([ "%d" % (c,) for c in board[h].tolist() ]),) for h in range(0, board.shape[0]) ]
  #return("""'[%s]'""" % (','.join(a),))

  #for h in range(0, c.shape[0]):
  #  for v in range(0, c.shape[1]):

def display_gameplay(element_id, boards, scores, timing):
  b_arr = []
  for board in boards:
    b = [ "[%s]" % (','.join([ "%d" % (c,) for c in board[h].tolist() ]),) for h in range(0, board.shape[0]) ]
    b_arr.append( "[%s]" % (','.join(b),) )
  s_arr = [ "%d" % (s,) for s in scores ]
  return("""<script type="text/javascript">display_gameplay("%s",[%s],[%s],%f)</script>""" % (element_id, ','.join(b_arr), ','.join(s_arr), timing))



javascript_base = """
<script type="text/Javascript">
var kernel = IPython.notebook.kernel;
function create_board(board_id, horizontal, vertical, n_colours) {
  //var board_id="#board_"+horizontal+"_"+vertical;
  if($(board_id).children().length==0) {
    console.log("Adding table to "+board_id);
    var trs=[];
    for(var v=0; v<vertical; v++) {
      var tds=[];
      for(var h=0; h<horizontal; h++) {
        tds.push("<td width='20' height='20' class='i_"+(horizontal-h-1)+"_"+(vertical-v-1)+"' h='"+(horizontal-h-1)+"' v='"+(vertical-v-1)+"'></td>");
      }
      trs.push("<tr>"+tds.join('')+"</tr>");
    }
    $(board_id)
      .empty()
      .append("<div class='score' style='width:"+(horizontal*20)+"px;text-align:center'>0</div>")
      .append("<table border='0' style='margin-top:0;'>"+trs.join('')+"</table>");
    $(board_id+' table')
      .off('click')
      .on('click', 'td', function(ev) {
        var cell = $(this);
        //console.log("Cell clicked : ", cell);
        var h=cell.attr('h') - 0;
        var v=cell.attr('v') - 0;
        console.log("Cell(h,v)=("+h+","+v+")");

        // https://github.com/fluxtream/fluxtream-ipy/blob/master/Communication%20between%20kernel%20and%20javascript%20in%20iPython%202.0.ipynb
        function handle_python_output(msg) {
          //console.log(msg);
          if( msg.msg_type == "error" ) {
            console.log("Javascript received Python error : ", msg.content);
          }
          else {  // execute_result
            var res_str = msg.content.data["text/plain"];
            //console.log("Javascript received Python Result : ", res_str);
            var res_json=res_str.replace(/[\\']/g,'"');  // NASTY kludge python->json
            //console.log("ie : ", res_json);
            var res=JSON.parse( res_json ); 
            //console.log("res = ", res);
            display_board(board_id, res.arr, undefined, res.score_inc);
          }
        }
        
        var cmd=[
          'board, score_inc, n_cols=crush.after_move(board, '+h+','+v+', '+n_colours+')',
          'arr=crush_ui.render_to_json(board)',
          'dict(arr=arr,score_inc=score_inc,n_cols=n_cols)'
        ].join(';');
        //console.log(cmd);
        
        kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});
      });
  }
  //$(board_id).append("<b>Hello</b>");
  //kernel.execute(command);
}
function display_board(board_id, matrix, score, score_inc) {
  console.log("display_board("+board_id+","+JSON.stringify(matrix).substr(0,37)+"... ,"++score+","+score_inc+")");
  var col=['#fff','#00f','#0f0','#f00','#666','#aaa'];
  if( typeof score == 'undefined' ) {
    score = $(board_id+' .score').text() |0;
    score += score_inc;
  }
  $(board_id+' .score').text(score | 0);
  matrix.forEach(function(row,h) {
      row.forEach(function(c,v) {
          //console.log(board_id+' .i_'+h+'_'+v);
          //$(board_id+' .i_'+h+'_'+v).html(c);
          $(board_id+' .i_'+h+'_'+v).css("background-color",col[c]);
        });
    });
}
function display_gameplay(board_id, boards, scores, timing) {
  function display_gameplay_step(board_i) {
    if(board_i<boards.length) {
      //console.log(boards[board_i]);
      display_board(board_id, boards[board_i], scores[board_i]);
      setTimeout( function() { display_gameplay_step(board_i+1); }, timing*1000);
    }
  }
  display_gameplay_step(0);
}
</script>
"""

javascript_test = """
<script type="text/Javascript">
var kernel = IPython.notebook.kernel;
function handle_python_output(msg) {
  console.log(msg);
  if( msg.msg_type == "error" ) {
    console.log("Javascript received Python error : ", msg.content);
  }
  else {
    var res = msg.content.data["text/plain"];
    console.log("Javascript received Python Result : ", res);
  }
}
var cmd='a=5+2;a+5';
kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});
</script>
"""

if __name__ == "__main__":
  import crush
  n_colours = 5
  b = crush.new_board(10,14,n_colours) # Same as portrait phone  1 screen~1k,  high-score~14k
  #print( display_via_javascript_script("#board_id", b) )
  print( render_to_json(b) )
  
