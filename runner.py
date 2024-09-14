import sys
import numpy as np
import pandas as pd

from cobaya.run import run
import emcee
import numpy as np

from cobaya.run import run
from source_code.likelihood import LSSlike
from time       import time
import sys

import warnings
warnings.filterwarnings('ignore')

from bios import read
from copy import deepcopy


from utils.utils import Suppressor,wbinned_setup

sys.path.append('../')
from samplers.samplers_interface import nautilus_interface,run_fisher


import pprint
pp = pprint.PrettyPrinter(indent=4)



info = read(sys.argv[1])
camb_path = '/users/chiaradeleo/Desktop/myeftcamb-main'
info['likelihood'] =  {'LSS': {'external': LSSlike,
                               'data_path': './mock_data/EFTflag_1_test_galonly_IApoly',
                               'debug_mode': False,
                               'camb_path': camb_path,
                               'use_noiseless_cls': True}}



info['force'] = True

if list(info['sampler'].keys())[0] == 'mcmc':
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
