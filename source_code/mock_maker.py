import sys,os,re
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sb

from IPython.display import display


from source_code.galdist import galaxy_distribution
from source_code.gwdist import gw_distribution

from source_code.fisher     import get_Fisher
from source_code.likelihood import LSSlike

from cobaya.run import run

from getdist.gaussian_mixtures import GaussianND
from copy import deepcopy
from getdist import plots,loadMCSamples,MCSamples

import matplotlib
from matplotlib import rc
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm

rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

red    = '#8e001c',

yellow = '#ffb302'

sidelegend = {'bbox_to_anchor': (1.04,0.5), 
              'loc': "center left",
              'frameon': False}

class MakeMock:

    def __init__(self,gwspecs,observables,camb_path,fiducial,test_parameter,generate_mock=True):

        self.Nbins_gal   = 10
        self.Nbins_gw    = gwspecs['Nbins_GW']
        self.observables = observables
        self.fiducial    = fiducial

        galspecs = {'fsky': 0.35,
                    'gal_per_arcmin': 30.,
                    'sigma_eps': 0.3}

        info = {'obs_settings': {'extra': None,
                                 'case': 'simple',
                                 'camb_path': camb_path,
                                 'calculation': 'CAMB'},

                'analysis_settings': {'lmin': 10,
                                      'lmax': 1500,
                                      'Nbin_ell': 20,
                                      'galaxy_specs': galspecs,
                                      'GW_specs': gwspecs}}

        info['sampler'] = {'Fisher': {'derivative': 'polynomial',
                                      'freepars': {test_parameter['name']: {'fiducial': self.fiducial[test_parameter['name']],
                                                                            'variation': test_parameter['variation'],
                                                                            'latex': test_parameter['latex']}},
                                      'fixedpars': {par: val for par,val in self.fiducial.items() if par != test_parameter['name']}}}


        if generate_mock:
            self.distributions = self.get_distributions()
            #save dist here
            self.fiducial_obs  = self.get_fiducial(info)
            self.covmats       = self.get_covmats()
            self.mock_data     = self.create_mock()
             

    def create_mock(self):

        theoryvec = self.fiducial_obs[self.all_cols]
        datavec = pd.DataFrame(columns=self.all_cols,index=theoryvec.index,dtype='float')
        errvec  = pd.DataFrame(columns=self.all_cols,index=theoryvec.index,dtype='float')
        for ellind,ell in enumerate(self.fiducial_obs['ells']):
            datavec.iloc[ellind] = np.random.multivariate_normal(theoryvec.iloc[ellind],self.covmats[ellind])
            errvec.iloc[ellind]  = np.sqrt(np.diag(self.covmats[ellind]))

        theoryvec['ells'] = self.fiducial_obs['ells']
        datavec['ells']   = self.fiducial_obs['ells']
        covmat_dict = {str(int(ell)): self.covmats[ellind] for ellind,ell in enumerate(self.fiducial_obs['ells'])}
        self.all_cols      = self.fishmodule.columns_ordering()  
        mock_dict = {'noiseless': theoryvec,
                     'noisy': datavec,
                     'covmat_dict': covmat_dict,
                     'all_cols': self.all_cols,
                     'distributions': self.distributions}

        return mock_dict


    def get_fiducial(self,fish_info):

        lmin = np.log10(fish_info['analysis_settings']['lmin'])
        lmax = np.log10(fish_info['analysis_settings']['lmax'])
        N    = fish_info['analysis_settings']['Nbin_ell']

        obs_settings = fish_info['obs_settings']
        if obs_settings['extra'] == None:
            obs_settings['extra'] = {}

        ell_lims = np.logspace(lmin,lmax,N) #creation of array-> N bin log spaced
        ells     = np.array([int(ell) for ell in 0.5*(ell_lims[:-1]+ell_lims[1:])])
        deltas   = (ell_lims[1:]-ell_lims[:-1]) #evaluation of the amplitude of each bin

        galaxy_specs  = fish_info['analysis_settings']['galaxy_specs']

        free_params = fish_info['sampler']['Fisher']['freepars']

        #MMmod: to be changed to work only with GW only?#####
        if 'GWWL' in self.distributions or 'GWC' in self.distributions:
            GW_specs = fish_info['analysis_settings']['GW_specs']
        else:
            GW_specs = {}

        fiducial_pars = {par: pardict['fiducial'] for par,pardict in free_params.items()} | fish_info['sampler']['Fisher']['fixedpars']
        dertype  = fish_info['sampler']['Fisher']['derivative']

        self.fishmodule = get_Fisher(fiducial_pars,free_params,ells,deltas,self.distributions,obs_settings,galaxy_specs,GW_specs,dertype)

        fiducial_obs = self.fishmodule.fidobs

        return fiducial_obs

    def get_covmats(self):

        covmat   = self.fishmodule.compute_covmat()
        self.all_cols = self.fishmodule.columns_ordering()

        covmats_by_ell = [self.fishmodule.pack_covmat(covmat,ellind,self.all_cols) for ellind,ell in enumerate(self.fishmodule.fidobs['ells'])]

        return covmats_by_ell


    def get_distributions(self):

        distributions = {}

        if 'GC' in self.observables:
            dist = galaxy_distribution('Euclid-{}'.format(self.Nbins_gal))
            bin_lims = dist.galdict['bin_lims']
            bin_mids = 0.5*(bin_lims[:-1]+bin_lims[1:])
            Nbins_gc = len(bin_lims)-1

            distributions['GC'] = {'dist': dist.galdict['binned_dist'],
                                   'Nbins': Nbins_gc,
                                   'zmean': bin_mids}

        if 'WL' in self.observables:
            dist = galaxy_distribution('Euclid-{}'.format(self.Nbins_gal))
            bin_lims = dist.galdict['bin_lims']
            bin_mids = 0.5*(bin_lims[:-1]+bin_lims[1:])
            Nbins_wl = len(bin_lims)-1

            distributions['WL'] = {'dist': dist.galdict['binned_dist'],
                                   'Nbins': Nbins_wl,
                                   'zmean': bin_mids}

        if 'GWC' in self.observables:

            gwdist  = gw_distribution('ET-{}'.format(self.Nbins_gw))


            bin_lims = gwdist.gwdict['bin_lims']
            bin_mids = 0.5*(bin_lims[:-1]+bin_lims[1:])
            Nbins_gwc = len(bin_lims)-1

            distributions['GWC'] = {'dist': gwdist.gwdict['binned_dist'],
                                    'Nbins': Nbins_gwc,
                                    'zmean': bin_mids}
        if 'GWWL' in self.observables:
            gwdist  = gw_distribution('ET-{}'.format(self.Nbins_gw))


            bin_lims = gwdist.gwdict['bin_lims']
            bin_mids = 0.5*(bin_lims[:-1]+bin_lims[1:])
            Nbins_gwl = len(bin_lims)-1

            distributions['GWWL'] = {'dist': gwdist.gwdict['binned_dist'],
                                     'Nbins': Nbins_gwl,
                                     'zmean': bin_mids}
            
        return distributions


