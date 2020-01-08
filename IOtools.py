import os 
import copy

# Func 1.1: write txt file ( append  or rewrite )
def txt_write(file_name,str,mode='a'):
    if not os.path.exists(file_name):
        mode = 'w'

    with open(file_name,mode) as f:
        f.write(str)

# get opt string as config
def get_config_str(opt):
    # begin
    config_str = '-'*60 +'\n'
    config_str += '---Configuration for %s---\n' %(opt['dataset'])
    # get config
    key_list = list(opt.keys())
    for key in key_list:
        config_str+= key+': '
        config_str+= str(opt[key]) + '\n'

    # end
    config_str += '-'*60 +'\n'


    return config_str
