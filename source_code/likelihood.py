import re
import numpy  as np
import sys
import pandas as pd
import re
from astropy import constants as const
import os


from cobaya.likelihood import Likelihood

from scipy.interpolate import interp1d

from source_code.compute_obs_sources import get_obs

from time import time

class LSSlike(Likelihood):

    def initialize(self):

        #Hard coded stuff
        
        self.feedback=self.debug_mode
        if self.use_noiseless_cls:
            print('Using noiseless Cls')
            self.data_Cls     = pd.read_csv(self.data_path+'_Cls_noiseless.dat',sep='\s+',header=0)
        else:
            self.data_Cls     = pd.read_csv(self.data_path+'_Cls_noisy.dat',sep='\s+',header=0)

        self.data_ells    = self.data_Cls['ells']
        self.covmat      = np.load(self.data_path+'_covmat.npy',allow_pickle=True).item()
        self.observables = np.load(self.data_path+'_source_distribution.npy',allow_pickle=True).item()
        self.obs_used = self.settings['obs_used']
        print('Observables used:', self.obs_used)
        cov_cut = {}
        invcov = {}
        '''CDL: First masking of the data. We generate a single dataset containing the full LSSxGW. Here we mask the data and the
        corresponding covariance elements that we do not want to use. E.g. if we want to use just the LSS part we drop all the columns (and
        rows in the covariance) that contain a GW observable: GWC, GWWL.'''
        for ell in self.data_ells:
            ell_str = str(int(ell))
            cov_df = self.covmat[ell_str]
            self.cols_to_drop = []
            if 'GWC' not in self.obs_used or ell>self.settings['scale_cut']['value']:   
                self.cols_to_drop = [col for col in cov_df.columns if 'WC' in col]
            if 'GWWL' not in self.obs_used or ell>self.settings['scale_cut']['value']:   
                self.cols_to_drop += [col for col in cov_df.columns if 'WL' in col]
            if 'GC' not in self.obs_used:
                self.cols_to_drop += [col for col in cov_df.columns if col.split('x')[0].startswith('G') or col.split('x')[1].startswith('G')]
            if 'WL' not in self.obs_used:
                self.cols_to_drop += [col for col in cov_df.columns if col.split('x')[0].startswith('L') or col.split('x')[1].startswith('L')]

            cov_df = cov_df.drop(columns=self.cols_to_drop)
            cov_df = cov_df.drop(index=self.cols_to_drop)
            cov_cut[ell_str] = cov_df
            invcov[ell_str] = pd.DataFrame(np.linalg.pinv(cov_df), index=cov_df.index, columns=cov_df.columns)

        self.invcov = invcov    
            


    def logp(self, **params_values):
        params = {key: value for key, value in params_values.items()}
       
        theory = get_obs(params, self.observables, self.data_ells, self.settings).Cls
        
  
       
        loglike = 0

        ell_diff = []
        chi2_per_ell = []
       
        for ind, ell in enumerate(self.data_ells):
            like_cols = self.invcov[str(int(ell))].columns
            gw_cols = [i for i, col in enumerate(like_cols) if 'W' in col]

            if len(like_cols) > 0:
                thvec = theory.iloc[ind][like_cols].values.copy()
                dtvec = self.data_Cls.iloc[ind][like_cols].values
                diffvec = thvec - dtvec
                '''CDL: Second masking or scale cut. Our code now doesn't allow to work with different ell_max for the LSS and GW part.
                Since the maximum multipole of LSS is expected to be higher then the one of GW, we generate the data for LSSxGW using the 
                ell_max of LSS (in this work is 1500) and then we cut all the GWs term (both auto and cross) that are at multipole higher than a
                value that can be chosen in the settings (in this work is 200).'''
                if ('scale_cut' in self.settings 
                    and self.settings['scale_cut'].get('method') == 'ell_cut_like' 
                    and ell > self.settings['scale_cut']['value']):
                    
                    diffvec[gw_cols] = 0.0

                chi2_val = np.dot(diffvec, np.dot(self.invcov[str(int(ell))], diffvec))
                loglike += -0.5 * chi2_val

            if self.feedback:
                diff_dict = {'ells': ell} | {col: thvec[i] - dtvec[i] for i, col in enumerate(like_cols)}
                ell_diff.append(pd.DataFrame(diff_dict, index=[0]))
                chi2_per_ell.append((ell, chi2_val))




        if self.feedback:
            print('Plotting')
            plot_df = pd.concat(ell_diff,ignore_index=True)
            plot_df = pd.melt(plot_df,id_vars=['ells'],value_vars=like_cols,var_name='Bin',value_name='value')
            import matplotlib.pyplot as plt
            import seaborn as sb
            plt.figure()
            plt.title('Theo-data')
            plt.axvline(x=200, color='gray', linestyle='--')
            sb.lineplot(plot_df,x='ells',y='value',hue='Bin',legend=0)
            plt.xscale('log')
            plt.show()

            chi2_df = pd.DataFrame(chi2_per_ell, columns=['ell', 'chi2'])

            plt.figure()
            plt.title('Chi²')
            plt.axvline(x=200, color='gray', linestyle='--')
            plt.plot(chi2_df['ell'], chi2_df['chi2'], marker='o', linestyle='-')
            plt.xscale('log')
            plt.ylabel('Chi²')
            plt.xlabel('ell')
            plt.tight_layout()
            plt.show()

        return loglike
