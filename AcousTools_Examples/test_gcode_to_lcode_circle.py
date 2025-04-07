from acoustools.Fabrication.Translator import gcode_to_lcode
from acoustools.Utilities import create_points

pth = 'acoustools/tests/data/gcode/circle.gcode'

pre_cmd = 'C0;\n'
post_cmd = 'C1;\nC3:10;\nC2;\n'

via = create_points(1,1,0,0,0.01)

pre_file = 'C5:gspat;#set solver\nC8;#set topbaord\nC6:20;#set iterations\n'
gcode_to_lcode(pth, pre_print_command=pre_cmd,post_print_command=post_cmd ,pre_commands=pre_file, max_stepsize=0.0001, 
               sig_type='Focal', travel_type='hypot', via=via, use_functions=True)
