
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
                //tds.push("<td width='20' height='20' class='i_"+(horizontal-h-1)+"_"+(vertical-v-1)+"'></td>");
                tds.push("<td width='20' height='20' h='"+(horizontal-h-1)+"' v='"+(vertical-v-1)+"'></td>");
            }
            trs.push("<tr>"+tds.join('')+"</tr>");
        }
        $(board_id).append("<table border=0>"+trs.join('')+"</table>");
        $(board_id+' table').click(function() {
            var cell = $(this).closest('td');
            console.log("Cell clicked : ", cell);
            var h=cell.attr('h') | 0;
            var v=cell.attr('v') | 0;
            console.log("Cell(h,v)=("+h+","+v+")");

            // https://github.com/fluxtream/fluxtream-ipy/blob/master/Communication%20between%20kernel%20and%20javascript%20in%20iPython%202.0.ipynb
            function handle_python_output(msg) {
                console.log(msg);
                if( msg.msg_type == "error") {
                  console.log("Javascript received Python error : ", msg.content);
                }
                else {  // execute_result
                  var res = msg.content.data["text/plain"];
                  console.log("Javascript received Python Result : ", res);
                  var arr = JSON.parse(res.arr);
                  
                }
            }
            
            var cmd='board, score, n_cols=crush.after_move(board, '+h+','+v+', '+n_colours+');a=crush.display_via_javascript_callback(board);dict(arr=a,score=score,n_cols=n_cols)';
            console.log(cmd);
            
            kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});
        });
    }
    //$(board_id).append("<b>Hello</b>");
    //kernel.execute(command);
}
function display_board(board_id,a) {
    var col=['#fff','#00f','#0f0','#f00','#666','#aaa'];
    a.forEach(function(ah,h) {
        ah.forEach(function(c,v) {
            //console.log(board_id+' .i_'+h+'_'+v);
            //$(board_id+' .i_'+h+'_'+v).html(c);
            $(board_id+' .i_'+h+'_'+v).css("background-color",col[c]);
        });
    });
}
</script>
"""

javascript_test = """
<script type="text/Javascript">
var kernel = IPython.notebook.kernel;
function handle_python_output(msg) {
    //console.log(msg);
    if( msg.msg_type == "error") {
      console.log("Javascript received Python error : ", msg.content);
    }
    else {
      var res = msg.content.data["text/plain"];
      console.log("Javascript received Python Result : ", res);
    }
}
var cmd='a=2+2;a+5';
kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});
</script>
"""
