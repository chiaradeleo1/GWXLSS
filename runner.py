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

info     = read(sys.argv[1])

#MMmod:
#creating output folder if it doesn't exist
directory = os.path.dirname(os.path.abspath(info['output']))

if not os.path.exists(directory):
    os.makedirs(directory)

if list(info['sampler'].keys())[0] == 'mcmc':
    #MMmod: this needs to be changed to be able to use multiple likelihoods at once
    likesets = deepcopy(info['likelihood']['LSS'])
    info['likelihood'] =  {'LSS': {'external': LSSlike,
                                   'data_path': likesets['data_path'],
                                   'debug_mode': likesets['debug_mode'],
                                   'camb_path': likesets['camb_path'],
                                   'use_noiseless_cls': likesets['use_noiseless_cls']}}
    info['sampler']['mcmc']['covmat'] = 'covariance_matrix.covmat'
    updated_info,sampler = run(info)
elif list(info['sampler'].keys())[0] == 'nautilus':
    nautilus = nautilus_interface(info)
elif list(info['sampler'].keys())[0] == 'Fisher':
    fishmat,info_dict  = run_fisher(info)
    fishmat.to_csv(info['output']+'_matrix.txt',sep='\t',header=True,index=False)
    np.save(info['output']+'_info.npy',info_dict)
else:
    sys.exit('Unknown sampling method: {}'.format(info['sampler']))
