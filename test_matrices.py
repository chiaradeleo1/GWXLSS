import sys,os
import numpy  as np
import pandas as pd
import numpy  as np

from source_code.likelihood import LSSlike

from time import time
from bios import read
from copy import deepcopy
from itertools import product
from tqdm import tqdm

from utils.utils import Suppressor,wbinned_setup

from samplers.samplers_interface import nautilus_interface,run_fisher

import warnings
warnings.filterwarnings('ignore')


import pprint
pp = pprint.PrettyPrinter(indent=4)

info     = read('settings/LCDM_fisher_3x2pt.yaml')
#pp.pprint(info)

models = ['LCDM']#,'MG']
theta_vec = [0.,0.001,0.01]

info['output'] = 'theta_test/LCDM_3x2pt_10bin'
dist_path = './dist_data/'
info['analysis_settings']['dist_path'] = dist_path+'MGflag_1_test_gal_source_distribution.npy'
info['obs_settings']['camb_path'] = '/home/Matteo/Codes/GW-MGCAMB'

for model in models:

    gal_info = deepcopy(info)
    gw_info  = deepcopy(info)

    if model == 'MG':
        gal_info['output'] = info['output']+'_MG'
        gal_info['sampler']['Fisher']['freepars']['mu0'] = {'fiducial': 0.64,
                                                            'variation': 0.1,
                                                            'latex': '\mu_0'}
        gal_info['sampler']['Fisher']['freepars']['sigma0'] = {'fiducial': 0.61,
                                                               'variation': 0.1,
                                                               'latex': '\Sigma_0'}

        gw_info['sampler']['Fisher']['freepars']['mu0'] = {'fiducial': 0.64,
                                                            'variation': 0.1,
                                                            'latex': '\mu_0'}
        gw_info['sampler']['Fisher']['freepars']['sigma0'] = {'fiducial': 0.61,
                                                               'variation': 0.1,
                                                               'latex': '\Sigma_0'}

    gal_info['sampler']['Fisher']['fixedpars']['b0_poly_GW'] = 0.830703
    gal_info['sampler']['Fisher']['fixedpars']['b1_poly_GW'] = 1.190547
    gal_info['sampler']['Fisher']['fixedpars']['b2_poly_GW'] = -0.928357
    gal_info['sampler']['Fisher']['fixedpars']['b3_poly_GW'] = 0.423292

    fishmat,info_dict  = run_fisher(gal_info)
    #print('')
    #print('3x2pt {}'.format(model))
    #pp.pprint(gal_info)
    fishmat.to_csv(gal_info['output']+'_matrix.txt',sep='\t',header=True,index=False)
    np.save(info['output']+'_info.npy',info_dict)

    for theta in theta_vec: 
        gw_info['output'] = info['output']+'_GW_theta{}'.format(theta)
        if model == 'MG':
            gw_info['output'] = gw_info['output']+'_MG'
        gw_info['analysis_settings']['dist_path'] = dist_path+'MGflag_1_test_galGWs_Nbin6_source_distribution.npy'
        gw_info['analysis_settings']['GW_specs'] = {'N_gw': int(1.e6),
                                                    'sigma_eps_gw': 0.01,
                                                    'fsky': 0.35,
                                                    'theta_min': theta}

        for i in range(4):
            gw_info['sampler']['Fisher']['freepars']['b{}_poly_GW'.format(i)] = gw_info['sampler']['Fisher']['freepars']['b{}_poly'.format(i)]

        try:
            print('')
            print('3x2pt+GW {}, theta={}'.format(model,theta))
            #pp.pprint(gw_info)
            fishmat,info_dict  = run_fisher(gw_info)
            fishmat.to_csv(gw_info['output']+'_matrix.txt',sep='\t',header=True,index=False)
            np.save(gw_info['output']+'_info.npy',info_dict)
        except:
            print('FAILED AT {} WITH THETA {}'.format(model,theta))

