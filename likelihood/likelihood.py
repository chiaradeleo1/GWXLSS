import re
import numpy  as np
import pandas as pd
from astropy import constants as const

from cobaya.likelihood import Likelihood

from scipy.interpolate import interp1d

from source_code_old.galdist     import galaxy_distribution
from source_code_old.gwdist      import gw_distribution
from source_code_old.compute_obs_Weyl import get_obs

from time import time

class LSSlike(Likelihood):

    def initialize(self):
        
        if self.use_obs['GW']:
            if self.datapath == None:
                gwdict       = gw_distribution().gwdict
                self.ni_gw       = [val for val in gwdict['binned_dist']]
                self.zbins_gw    = gwdict['bin_lims']
                self.ells     = np.logspace(1,np.log10(4000),20)
            else:
                gwdict           = np.load(self.datapath+'_gwdist.npy',allow_pickle=True).item()
                self.ni_gw           = [val for val in gwdict['binned_dist']]
                self.zbins_gw        = gwdict['bin_lims']
                if self.use_obs['GC']==False and self.use_obs['WL']==False: #(GW only case)
                    self.data_Cls     = pd.read_csv(self.datapath+'_Cls.dat',sep='\s+',header=0)
                    self.data_ells    = self.data_Cls['ells']
                        
                    self.covmat = np.load(self.datapath+'_covmat.npy',allow_pickle=True).item()
                    tini = time()
                    #inversion of covmat
                    self.invcov = {key: np.linalg.pinv(self.covmat[key]) for key in self.covmat}
                    print('Covmats inverted in {:.3f}'.format(time()-tini))

            self.z_win = np.linspace(0.001,4.,100)
            self.k_max_Boltzmann = 10
        else:
            gwdict = None
            self.ni_gw = None
            self.zbins_gw = None
            self.ells_gw = None
            
        if self.use_obs['GC'] or self.use_obs['WL']:
            if self.datapath == None:
                galdict       = galaxy_distribution().galdict
                self.ni       = [val for val in galdict['binned_dist']]
                self.ells     = np.logspace(1,np.log10(4000),20)
            else:
                galdict           = np.load(self.datapath+'_galdist.npy',allow_pickle=True).item()
                self.ni           = [val for val in galdict['binned_dist']]
                self.lumfunc      = galdict['luminosity']
                self.data_Cls     = pd.read_csv(self.datapath+'_Cls.dat',sep='\s+',header=0)
                self.data_ells    = self.data_Cls['ells']
                self.covmat = np.load(self.datapath+'_covmat.npy',allow_pickle=True).item()
                tini = time()
                self.invcov = {key: np.linalg.pinv(self.covmat[key]) for key in self.covmat}
                print('Covmats inverted in {:.3f}'.format(time()-tini))
                
        else:
            galdict= None
            self.ni= self.ni_gw
            self.zbins = self.zbins_gw
            #self.ells= self.ells_gw
    
        self.z_camb = np.linspace(0.001,4.,100)
        self.z_win  = np.logspace(-3,np.log10(4),500)
        self.k_max_Boltzmann = 10
        
        
    def get_requirements(self):


        return {'omegam': None,
                'Pk_interpolator':
                {'z': self.z_camb,
                 'k_max': self.k_max_Boltzmann,
                 'nonlinear': [False,True] ,
                 'vars_pairs': ([['delta_tot',
                                  'delta_tot'],
                                 ['Weyl',
                                  'Weyl']])},
                'comoving_radial_distance': {'z': self.z_camb},
                'angular_diameter_distance': {'z': self.z_camb},
                'Hubble': {'z': self.z_camb, 'units': 'km/s/Mpc'}}



    def get_cosmo_dict(self,provider,params): #definition of dictionary
        cosmo_dict = {'z': self.z_win,
                      'Omm': provider.get_param('omegam'),
                      'H0': provider.get_param('H0'),
                      'As': provider.get_param('As'),
                      'ns': provider.get_param('ns'),
                      'bias': interp1d(self.z_win,[sum([params['b{}_poly'.format(ind)]*np.power(z,ind) for ind in range(4)]) for z in self.z_win]),
                      'H0_Mpc': provider.get_param('H0')/const.c.to('km/s').value,
                      'comov_dist': interp1d(self.z_camb,provider.get_comoving_radial_distance(self.z_camb)),
                      'angular_dist': interp1d(self.z_camb,provider.get_angular_diameter_distance(self.z_camb)),
                      'H_Mpc': interp1d(self.z_camb,provider.get_Hubble(self.z_camb, units='1/Mpc')),
                      'Pk_linear': provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=False,
                                                               extrap_kmin=self.k_min_extrap,
                                                               extrap_kmax=self.k_max_extrap),
                      'Pk_delta': provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=True,
                                                               extrap_kmin=self.k_min_extrap,
                                                               extrap_kmax=self.k_max_extrap),
                      'Pk_Weyl': provider.get_Pk_interpolator(('Weyl', 'Weyl'), nonlinear=False,
                                                               extrap_kmin=self.k_min_extrap,
                                                               extrap_kmax=self.k_max_extrap)}



        ks = 0.001
        P_z_k = cosmo_dict['Pk_delta'].P(cosmo_dict['z'], ks)
        cosmo_dict['Dz'] = np.sqrt(P_z_k / cosmo_dict['Pk_delta'].P(0.001, ks))

        cosmo_dict['IA_term'] = interp1d(cosmo_dict['z'], -params['A_IA']*0.0134*cosmo_dict['Omm']*(1+cosmo_dict['z'])**params['eta_IA']/cosmo_dict['Dz'])

        return cosmo_dict


    def logp(self, **params_values):

        ells = np.linspace(100,1000,10)
        cosmo_dict = self.get_cosmo_dict(self.provider,params_values)

        if self.datapath == None:
            self.obs   = get_obs(self.ni,self.ni_gw,cosmo_dict,self.use_obs,self.ells,use_Weyl=self.use_Weyl)
            loglike = -np.inf
        else:
            self.obs   = get_obs(self.ni,self.ni_gw,cosmo_dict,self.use_obs,self.data_Cls['ells'],use_Weyl=self.use_Weyl)
            loglike = 0
            for ind,ell in enumerate(self.data_ells):
                dof = len(self.data_Cls.columns)-1
                diffvec = np.array([self.data_Cls.iloc[ind][col]-self.obs.Cls.iloc[ind][col] for col in self.data_Cls.columns if col != 'ells'])
#calculation of logarithmic likelihoos
                loglike += -0.5*np.dot(diffvec,np.dot(self.invcov[str(int(ell))],diffvec))
                
        return loglike