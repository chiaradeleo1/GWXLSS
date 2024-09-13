import numpy as np

from cobaya.run import run
from source_code.likelihood import LSSlike
from time       import time
import sys

import warnings
warnings.filterwarnings('ignore')

use_fiducial = False
#camb_path = '/Users/chiaradeleo/myenv/lib/python3.12/site-packages'
camb_path= '/Users/chiaradeleo/Desktop/myeftcamb-main'

fiducial = {'ombh2': 0.022445,
            'omch2': 0.1205579307,
            'ns': 0.96,
            'As': 2.12605e-09,
            'tau': 0.05,
            'H0': 67.,
            'w': -1.,
            'wa': 0.,
            'mnu': 0.06,
            'a0': - 0.007589,
            'a1' :  0.002008,
            'a2' : - 0.004127,
            'a3' :  0.002918,
            'a4' : -0.0006784,
            'b0_poly': 0.830703,
            'b1_poly': 1.190547,
            'b2_poly': -0.928357,
            'b3_poly': 0.423292}

fiducial['logA'] = np.log(fiducial.pop('As')*1.e+10)

info = {'sampler': {'mcmc': {'max_tries':100000}},
                             #'covmat': 'LCDM_covmat_3x2pt.covmat'}},
        'likelihood': {'LSS': {'external': LSSlike,
                               'data_path': './mock_data/LCDM_test_galonly_IApoly',
                               'debug_mode': True,
                               'camb_path': camb_path,
                               'use_noiseless_cls': True}}}


info['params'] = {'ombh2': {'latex': '\Omega_\mathrm{b} h^2',
                            'prior': {'min': 0.005,'max': 0.1},
                            'proposal': 0.0001,
                            'ref':  0.0224},
                  'omch2': {'latex': '\Omega_\mathrm{c} h^2',
                            'prior': {'min': 0.001,'max': 0.99},
                            'proposal': 0.0005,
                            'ref':  0.12},
                  'ns': {'latex': 'n_\mathrm{s}',
                         'prior': {'min': 0.6,'max': 1.2},
                         'proposal': 0.002,
                         'ref':  0.96},
                  'logA': {'drop': True,
                           'latex': '\log(10^{10} A_\mathrm{s})',
                           'prior': {'max': 7.0,'min': 1.6},
                           'proposal': 0.001,
                           'ref': 3.05},
                  'As': {'latex': 'A_\mathrm{s}',
                         'value': 'lambda logA: 1e-10*np.exp(logA)'},
                  'tau': 0.05,
                  'H0': {'latex': 'H_0',
                         'prior': {'max': 100.0,'min': 40.0},
                         'proposal': 0.5,
                         'ref':  67.0},
                  'w': -1.,
                  'wa': 0.,
                  'mnu': 0.06,
                  'omegab': {'latex': '\Omega_\mathrm{b}'},
                  'omegam': {'latex': '\Omega_\mathrm{m}'},
                  'sigma8': {'latex': '\sigma_8'},
                  'a0': - 0.007589,
                  'a1' :  0.002008,
                  'a2' : - 0.004127,
                  'a3' :  0.002918,
                  'a4' : -0.0006784,
                  'b0_poly': 0.830703,
                  'b1_poly': 1.190547,
                  'b2_poly': -0.928357,
                  'b3_poly': 0.423292}
if use_fiducial:
    info['sampler'] = {'evaluate': {'override': {k:v for k,v in fiducial.items() if type(info['params'][k]) == dict}}}
else:
    info['sampler'] = {'evaluate': None}

updated_info, sampler = run(info)
