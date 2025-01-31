import sys,os,re
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sb
from source_code.galdist import galaxy_distribution
from source_code.gwdist import gw_distribution

from itertools import product
from copy      import deepcopy
from time      import time

from scipy.interpolate import interp1d
from scipy.integrate   import trapz

import warnings
warnings.filterwarnings('ignore')

from samplers.samplers_interface import nautilus_interface,run_fisher

fiducial = {'ombh2': 0.022445,
            'omch2': 0.1205579307,
            'ns': 0.96,
            'As': 2.12605e-09,
            'H0': 67.,
            'w': -1.,
            'wa': 0.,
            'tau': 0.05,
            'mnu': 0.06,
            'a0': - 0.007589,
            'a1' :  0.002008,
            'a2' : - 0.004127,
            'a3' :  0.002918,
            'a4' : -0.0006784,
            'omegab': 0.05,
            'sigma8': 0.84,
            'omegam' : 0.31,
            #'A_IA': 1.72,
            #'eta_IA': -0.41,
            'b0_poly': 0.830703,
            'b1_poly': 1.190547,
            'b2_poly': -0.928357,
            'b3_poly': 0.423292,
            'MG_flag': 1,
            'pure_MG_flag': 2,
            'musigma_par': 1,
            'DE_model': 0,
            'mu0': 0.64,
            'sigma0': 0.61,
           }
fiducial['logA'] = np.log(fiducial['As']*1.e+10)

config_dict = {"output": "chains/muSigmaCDM_fisher_3x2pt_GWcounts_fixednuis",
               "obs_settings": {"extra": None,
                                "case": "simple",
                                "camb_path": "/home/Matteo/Codes/GW-MGCAMB",
                                "calculation": "CAMB",},
                "analysis_settings":{"dist_path": "./mock_data/MGflag_1_test_galGWC_source_distribution.npy",
                                    "lmin": 10,
                                    "lmax": 1500,
                                    "Nbin_ell": 20,
                                    "galaxy_specs": {"fsky": 0.35,
                                                    "gal_per_arcmin": 30.0,
                                                    "sigma_eps": 0.3,},
                                    "GW_specs": {"fsky": 0.35,
                                                "N_gw": 100000,
                                                "sigma_eps_gw": 0.005,},},
                "sampler": { "Fisher": {"derivative": "2PT",
                                        "freepars": {"ombh2": {"fiducial": 0.022445,
                                                                "variation": 0.1,
                                                                "latex": r"\Omega_{\rm b}\,h^2",},
                                                    "omch2": {"fiducial": 0.1205579307,
                                                        "variation": 0.1,
                                                        "latex": r"\Omega_{\rm c}\,h^2",},
                                                    "H0": {"fiducial": 67.0,
                                                        "variation": 0.1,
                                                        "latex": r"H_0",},
                                                    "ns": {"fiducial": 0.96,
                                                        "variation": 0.1,
                                                        "latex": r"n_{\rm s}",},
                                                    "logA": {"fiducial": 3.05685,
                                                        "variation": 0.1,
                                                        "latex": r"\log{10^{10}\,A_{\rm s}",},
                                                    "mu0": {"fiducial": -1.23,
                                                            "variation": 0.1,
                                                            "latex": r"\mu_0",},
                                                    "sigma0": {"fiducial": -0.17,
                                                            "variation": 0.1,
                                                            "latex": r"\sigma_0",},},
                "fixedpars":{"tau": 0.05,
                            "a0": -0.007589,
                            "a1": 0.002008,
                            "a2": -0.004127,
                            "a3": 0.002918,
                            "a4": -0.0006784,
                            "b0_poly": 0.830703,
                            "b1_poly": 1.190547,
                            "b2_poly": -0.928357,
                            "b3_poly": 0.423292,
                            "w": -1.0,
                            "wa": 0.0,
                            "mnu": 0.06,
            },
        }
    }
}

#N_gw_configurations = [1e4, 5*1e4, 1e5, 5*1e5, 1e6]
#sigma_dL_configurations = [0.1, 0.05, 0.01, 0.005]
N_bins_configurations = [1,2,5,10]
obs = ['GWC', 'GWWL', 'GWs']
output_path=[]

# for N_gw, sigma_dL, obs  in product(N_gw_configurations, sigma_dL_configurations , obs):
#     config_dict["output"] = f"chains_MG/muSigmaCDM_fisher_3x2pt_{obs}_fixednuis_Ngw{int(N_gw)}_sigma{sigma_dL}"
#     config_dict["analysis_settings"]["dist_path"] = f"./mock_data/MGflag_1_test_gal{obs}_source_distribution.npy"
#     config_dict["analysis_settings"]["GW_specs"]["N_gw"] = N_gw
#     config_dict["analysis_settings"]["GW_specs"]["sigma_eps_gw"] = sigma_dL
#     output_path.append(config_dict["output"])
#     cases.append(r"$N_{gw}$={int(N_gw)}, $\sigma$={sigma_dL}, LSS $\times$ {obs}")

#     info = config_dict

#     fishmat, info_dict = run_fisher(info)
        
#     fishmat.to_csv(info['output'] + '_matrix.txt', sep='\t', header=True, index=False)
        
        
#     np.save(info['output'] + '_info.npy', info_dict)
#     print("Fisher analysis completed for ", info['output'])

for N_bins, obs  in product(N_bins_configurations , obs):
    config_dict["output"] = f"chains_MG/muSigmaCDM_fisher_3x2pt_{obs}_Ngwbins{int(N_bins)}"
    if N_bins == 10:
        config_dict["analysis_settings"]["dist_path"] = f"./mock_data/MGflag_1_test_gal{obs}_source_distribution.npy"
    else:
        if obs == 'GWs':
            config_dict["analysis_settings"]["dist_path"] = f"./mock_data/MGflag_1_test_gal{obs}_Nbin{int(N_bins)}_source_distribution.npy"
        else:
            config_dict["analysis_settings"]["dist_path"] = f"./mock_data/MGflag_1_test_gal{obs}GWs_Nbin{int(N_bins)}_source_distribution.npy"
    output_path.append(config_dict["output"])
    
    info = config_dict

    fishmat, info_dict = run_fisher(info)
        
    fishmat.to_csv(info['output'] + '_matrix.txt', sep='\t', header=True, index=False)
        
        
    np.save(info['output'] + '_info.npy', info_dict)
    print("Fisher analysis completed for ", info['output'])

print("All Fisher analyses completed")
