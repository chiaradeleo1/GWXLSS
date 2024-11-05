import camb
import numpy as np
from source_code.compute_obs_sources import get_obs

observables=np.load('./mock_data/LCDM_test_galGW_source_distribution.npy',allow_pickle=True).item()

cosmo_pars = {'ombh2': 0.022445,
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
            #'A_IA': 1.72,
            #'eta_IA': -0.41,
            'b0_poly': 0.830703,
            'b1_poly': 1.190547,
            'b2_poly': -0.928357,
            'b3_poly': 0.423292,
            'MG_flag':0}

extra={'MG_flag':0}

settings={'camb_path': camb.__path__,
         'case': 'simple',
         'calculation': 'CAMB',
          'extra':extra}
analysis_settings = {'Nbin_ell': 20,
                     'lmin': 10,
                     'lmax': 1500}
lmin = np.log10(analysis_settings['lmin'])
lmax = np.log10(analysis_settings['lmax'])
N    = analysis_settings['Nbin_ell']

ell_lims = np.logspace(lmin,lmax,N) #creation of array-> N bin log spaced
ells     = np.array([int(ell) for ell in 0.5*(ell_lims[:-1]+ell_lims[1:])]) 
#evaluation of middle points of each bin 
deltas   = (ell_lims[1:]-ell_lims[:-1]) #evaluation of the amplitude of each bin
calc_obs = get_obs(cosmo_pars,observables,ells,settings,feedback=True)
#print(calc_obs.Cls)
