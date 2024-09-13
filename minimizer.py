import sys
import numpy as np
import pandas as pd

from cobaya.run import run
import emcee

from copy import deepcopy
from bios import read

from likelihood.likelihood import fakeDESI
from likelihood.EFT_priors import EFT_priors

from utils.utils import Suppressor

info = read(sys.argv[1])

info['likelihood'] = {'F-DESI': {'external': fakeDESI,
                                 'data_path': './mock_data/DESI_table',
                                 'debug_mode': True,
                                 'use_calibration': True}}
if info['BBN_prior']:
    info['params']['ombh2']['prior'] = {'dist': 'norm',
                                        'loc': 0.02218,
                                        'scale': 0.00055}

if 'conditions' in info:
    if info['conditions'] == 'only':
        info['likelihood'] = {'EFTprior': {'external': EFT_priors,
                                           'conditions': 'full'}}
        del info['params']['rdh']
    else:
        info['likelihood']['EFTprior'] = {'external': EFT_priors,
                                          'conditions': info['conditions']}

    del info['conditions']

info['force'] = True

info['sampler'] = {'minimize': None}

if 'wsettings' in info:
    for node in range(len(info['theory']['camb']['extra_args']['znodes'])):
        info['params']['w_{}'.format(node)] = deepcopy(info['wsettings'])
        info['params']['w_{}'.format(node)]['latex'] = 'w_{}'.format(node)

updated_info,sampler = run(info)
