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


info     = read('settings/LCDM_fisher_3x2pt+GW.yaml')
info['output'] = 'LCDM_LSS_GW'
N_gw = [1e5,1e6,1e7,1e8]
l_cut = [100, 200, 1000, 1500]
sigma_dL = [0.001, 0.005, 0.01, 0.05, 0.1]
cases_list = ['density', 'redshift', 'lensing', 'velocity', 'lsd', 'evolve', 'potential', 'gradpotential', 'ISW']
cases = []
failed = []
for case, N_gw, l_cut, sigma_dL in product(cases_list, N_gw, l_cut, sigma_dL):
    try:
        if case not in cases:
            cases.append(case)
        info['obs_settings']['case'] = cases
        info['analysis_settings']['gw_specs']['N_gw'] = N_gw
        info['analysis_settings']['gw_specs']['sigma_eps_gw'] = sigma_dL
        info['analysis_settings']['gw_specs']['scale_cut']['value'] = l_cut
        info['output'] = info['output']+str(N_gw)+str(sigma_dL)+str(l_cut)+str(case)
        fishmat,info_dict  = run_fisher(info)
        fishmat.to_csv(info['output']+'_matrix.txt',sep='\t',header=True,index=False)
        np.save(info['output']+'_info.npy',info_dict)
    except Exception as e:
        failed_entry = f"FAILED: case={case}, N_gw={N_gw}, l_cut={l_cut}, sigma_dL={sigma_dL}, error={str(e)}\n"
        failed.append(failed_entry)
        with open('failed_cases.txt', 'w') as f:
            f.writelines(failed)



