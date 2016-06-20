
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
                tds.push("<td width='20' height='20' class='i_"+(horizontal-h-1)+"_"+(vertical-v-1)+"'></td>");
            }
            trs.push("<tr>"+tds.join('')+"</tr>");
        }
        $(board_id).append("<table border=0>"+trs.join('')+"</table>");
        $(board_id).click(function(e) {
            console.log("Cell clicked : ",e);
            
            function handle_output(out_type, out) {
                console.log(out_type);
                console.log(out);
                var res = null;
                 // if output is a print statement
                if(out_type == "stream"){
                    res = out.data;
                }
                // if output is a python object
                else if(out_type === "pyout"){
                    res = out.data["text/plain"];
                }
                // if output is a python error
                else if(out_type == "pyerr"){
                    res = out.ename + ": " + out.evalue;
                }
                // if output is something we haven't thought of
                else{
                    res = "[out type not implemented]";   
                }
                document.getElementById("result_output").value = res;
            }
            
            var cmd1='board, score, n_cols=crush.after_move(board, 0,0, '+n_colours+')';
            console.log(cmd1);
            
            kernel.execute(cmd1, {'output' : function(out_type, out) {
                    var cmd2='crush.display_via_javascript_callback(board)';

                    kernel.execute(cmd2, {'output' : handle_output}, {silent:false});
                    console.log(cmd2);

                    //var html_cmd2 = 'HTML('+cmd2+')';
                    //kernel.execute(html_cmd2);
                    //console.log(html_cmd2);
                }}, {silent:false});
            
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

