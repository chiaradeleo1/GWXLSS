import sys,os
import numpy  as np
import pandas as pd
import numpy  as np

from source_code.likelihood import LSSlike

from time import time
from bios import read
from copy import deepcopy

from utils.utils import Suppressor,wbinned_setup

#sys.path.append('../')
from cobaya.run                  import run
from samplers.samplers_interface import nautilus_interface,run_fisher

import warnings
warnings.filterwarnings('ignore')


import pprint
pp = pprint.PrettyPrinter(indent=4)

if len(sys.argv) < 2:
    sys.exit('\033[1;31m' + 'ERROR! MISSING ARGUMENT! '  + 
             f"You must provide a settings file. Example:\n"
             f"python runner.py settings_filename.yaml"+ '\033[0m')
info = read(sys.argv[1])

#MMmod: this needs to be changed to be able to use multiple likelihoods at once
if list(info['sampler'].keys())[0] in ['mcmc','nautilus','evaluate']:
    likesets = deepcopy(info['likelihood']['LSS'])
    info['likelihood'] =  {'LSS': {'external': LSSlike,
                                   'data_path': likesets['data_path'],
                                   'debug_mode': likesets['debug_mode'],
                                   'settings': likesets['settings'],
                                   'use_noiseless_cls': likesets['use_noiseless_cls']}}

#MMmod:
#creating output folder if it doesn't exist
if 'output' in info:
    directory = os.path.dirname(os.path.abspath(info['output']))

    if not os.path.exists(directory):
        os.makedirs(directory)

if list(info['sampler'].keys())[0] == 'mcmc':
    updated_info,sampler = run(info)
elif list(info['sampler'].keys())[0] == 'evaluate':
    info['sampler'] = {'evaluate': {'override': {k: v["ref"]["loc"] for k,v in info['params'].items() if isinstance(v, dict) and "ref" in v and "loc" in v["ref"]}}}
    updated_info,sampler = run(info)
elif list(info['sampler'].keys())[0] == 'nautilus':
    nautilus = nautilus_interface(info)
elif list(info['sampler'].keys())[0] == 'Fisher':
    print('RUNNING FISHER MATRIX CODE')
    fishmat,info_dict  = run_fisher(info)
    fishmat.to_csv(info['output']+'_matrix.txt',sep='\t',header=True,index=False)
    np.save(info['output']+'_info.npy',info_dict)
else:
    sys.exit('Unknown sampling method: {}'.format(info['sampler']))
