import re
import numpy  as np
import sys
import pandas as pd
import re
from astropy import constants as const

from cobaya.likelihood import Likelihood

from scipy.interpolate import interp1d

from source_code.compute_obs_sources import get_obs

from time import time


class LSSlike(Likelihood):

    def initialize(self):

        #Hard coded stuff
        
        self.feedback=self.debug_mode
  
        if self.use_noiseless_cls:
            self.data_Cls     = pd.read_csv(self.data_path+'_Cls_noiseless.dat',sep='\s+',header=0)
        else:
            self.data_Cls     = pd.read_csv(self.data_path+'_Cls_noisy.dat',sep='\s+',header=0)

        self.data_ells    = self.data_Cls['ells']
        self.covmat      = np.load(self.data_path+'_covmat.npy',allow_pickle=True).item()
        self.observables = np.load(self.data_path+'_source_distribution.npy',allow_pickle=True).item()
        

        if not any('GW' in key for key in self.obs_used):

            
            self.cols_to_drop = [col for col in self.data_Cls.columns if 'W' in col]
            self.rows_to_drop = self.cols_to_drop.copy()

            for ell in self.data_ells:
                ell_str = str(int(ell)) 
                cov_df = self.covmat[ell_str]
                

                cov_df = cov_df.drop(columns=self.cols_to_drop)
                cov_df = cov_df.drop(index=self.rows_to_drop)
                self.covmat[ell_str] = cov_df 
            self.data_Cls = self.data_Cls.drop(columns=self.cols_to_drop)

        elif 'GC' or 'WL' not in self.obs_used:
            
            self.cols_to_drop = [col for col in self.data_Cls.columns if col.startswith('L') or col.startswith('G') or 'xL' in col or 'xG' in col]
            self.rows_to_drop = self.cols_to_drop.copy()
                
            for ell in self.data_ells:
                ell_str = str(int(ell)) 
                cov_df = self.covmat[ell_str]

                
                cov_df = cov_df.drop(columns=self.cols_to_drop)
                cov_df = cov_df.drop(index=self.rows_to_drop)
                self.covmat[ell_str] = cov_df
                
            self.data_Cls = self.data_Cls.drop(columns=self.cols_to_drop)
        

                


        tini = time()
        #inversion of covmat
        self.invcov = {key: np.linalg.pinv(cov) for key,cov in self.covmat.items()}
        print('Covmats inverted in {:.3f}'.format(time()-tini))

    


    def logp(self, **params_values):
        params = {key: value for key, value in params_values.items()}
        theory = get_obs(params, self.observables, self.data_ells, self.settings).Cls
       
        loglike = 0
        like_cols = [col for col in self.data_Cls.columns if col != 'ells']
        ell_diff = []
        chi2_per_ell = []
        gw_cols = [col for col in like_cols if 'WC' in col or 'WL' in col]
        
       
        for ind, ell in enumerate(self.data_ells):

            thvec = theory.iloc[ind][like_cols].values.copy()
            dtvec = self.data_Cls.iloc[ind][like_cols].values
            if ('scale_cut' in self.settings and self.settings['scale_cut'].get('method') == 'ell_cut_like'):
                if ell > self.settings['scale_cut']['value']:
                    for i, col in enumerate(like_cols):
                        if col in gw_cols:
                            thvec[i] = dtvec[i]


            diffvec = thvec - dtvec
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
