import re
import numpy  as np
import sys
import pandas as pd
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
        
        tini = time()
        #inversion of covmat
        self.invcov = {key: np.linalg.pinv(cov) for key,cov in self.covmat.items()}
        print('Covmats inverted in {:.3f}'.format(time()-tini))

    


    def logp(self, **params_values):
        params = {key: value for key, value in params_values.items() }
        
        #self.obs = get_obs(params,self.observables, self.data_ells, self.settings,feedback=self.feedback)
        theory = get_obs(params,self.observables, self.data_ells, self.settings).Cls
        
        loglike = 0
        like_cols = [col for col in self.data_Cls.columns if col != 'ells']
        ell_diff = []
        for ind,ell in enumerate(self.data_ells):
            thvec   = theory.iloc[ind][like_cols].values
            dtvec   = self.data_Cls.iloc[ind][like_cols].values
            diffvec = thvec-dtvec 
            loglike += -0.5*np.dot(diffvec,np.dot(self.invcov[str(int(ell))],diffvec))
            if self.feedback: 
                df = pd.DataFrame({'ells': ell}|{col: theory.iloc[ind][col]-self.data_Cls.iloc[ind][col] for col in like_cols},index=[0])
                ell_diff.append(df)


        if self.feedback:
            plot_df = pd.concat(ell_diff,ignore_index=True)
            plot_df = pd.melt(plot_df,id_vars=['ells'],value_vars=like_cols,var_name='Bin',value_name='value')
            import matplotlib.pyplot as plt
            import seaborn as sb
            plt.figure()
            sb.lineplot(plot_df,x='ells',y='value',hue='Bin',legend=0)
            plt.xscale('log')
            plt.show()

        return loglike
