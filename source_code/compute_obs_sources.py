import sys
import numpy as np
import pandas as pd

import warnings
import pandas as pd

# Suppress the specific PerformanceWarning from Pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from scipy.interpolate import interp1d
from scipy.integrate   import trapz

from copy      import deepcopy
from itertools import product
from time      import time

possible_observables = ['GC','WL','GWC', 'GWWL' ]
cases = {'redshift': False,
         'lensing': False,
         'velocity': False,
         'potential':False,
         'lsd': False,
         'evolve': False,
         'potential': False,
         'gradpotential': False,
         'ISW': False,
         'SW': False,
         'volume': False}
class get_obs:

    def __init__(self,params, observables,ells, settings, feedback=False):

        self.observables = observables
        self.cases       = deepcopy(cases)
        self.params      = deepcopy(params)
        if 'logA' in self.params:
            self.params['As'] = np.exp(self.params.pop('logA'))*1.e-10
   
                
        self.extra_params = settings['extra']
        for case in settings.get('cases') or []:
            if case in self.cases:
                self.cases[case] = True
            else:
                sys.exit('\033[1;31m' + f'ERROR! UNKNOWN CASE: {case}' + '\033[0m' + f"\nPossible cases: {', '.join(self.cases.keys())}")
  
                
        self.camb_path    = settings['camb_path']
        self.ells         = ells
        self.obs_used    = settings['obs_used']
        

        self.zinterps        = np.logspace(-3,np.log10(5),500)
        self.k_max_Boltzmann = 10

        #MMmod: switch here if we want to customize
        self.params['dark_energy_model'] = 'ppf'
        
        for obs in self.observables.keys():
            if obs not in possible_observables:
                sys.exit('\033[1;31m' + f"ERROR! Unknown observable: {obs}. Possible: GC, WL, GWWL, GWC" + '\033[0m')

        self.feedback     = feedback
                

        tini = time()
        self.Cls = self.get_cls(ells)
        tend = time()
        if self.feedback:
            print('\033[1;32m' + f'Cls computed in {tend - tini:.1f} s' + '\033[0m')




    

    def get_cosmo_dict(self,params): 

        #CDL: This is hardcoded, can we make it automatic? These are the params that are not recognized by CAMB        
        nuis_keys  = ['_derived','a0','a1', 'a2', 'a3', 'a4', 'b0_poly', 'b1_poly', 'b2_poly', 'b3_poly', 
                      'b0_poly_GW', 'b1_poly_GW', 'b2_poly_GW', 'b3_poly_GW','logA', 'omegam', 'omegab', 'sigma8']
        if self.extra_params:
            nuis_keys = nuis_keys + list(self.extra_params.keys())
        
        cosmo_params = {par: params[par] for par in params.keys() if par not in nuis_keys}
        cosmo_dict = {'cosmo_params': cosmo_params}
        
            
        IA =[sum([params['a{}'.format(ind)]*np.power(z,ind) for ind in range(5)]) for z in self.zinterps] 
        cosmo_dict['IA_term']= interp1d(self.zinterps,IA)

        bgrid = [sum([params['b{}_poly'.format(ind)]*np.power(z,ind) for ind in range(4)]) for z in self.zinterps]
        cosmo_dict['bias'] = interp1d(self.zinterps,bgrid)

        if 'GWC' in self.observables:
            bgrid_GW = [sum([params['b{}_poly_GW'.format(ind)]*np.power(z,ind) for ind in range(4)]) for z in self.zinterps]
            cosmo_dict['bias_GW'] = interp1d(self.zinterps,bgrid_GW)

        return cosmo_dict


    def get_source_Cls(self,cosmo):

        sys.path.insert(0,self.camb_path)
        import camb
        from   camb.sources import SplinedSourceWindow
        from   camb         import model
        self.z            = np.linspace(0.001,4,500)
        
        cosmo_pars = cosmo['cosmo_params']
        MGCAMB_flag = {'MG_flag', 'musigma_par', 'pure_MG_flag', 'DE_model'}
        for flag in MGCAMB_flag:
            if flag in cosmo_pars:
                cosmo_pars[flag] = int(cosmo_pars[flag]) #CDL: CAMB expects these to be integers, but they are read
                                                         #as floats from the lieklihood, so we need to convert them
        pars = camb.set_params(**cosmo_pars)
        pars = self.set_camb_specs(pars) #This sets the cases for CAMB (Limber & friends)
        pars.NonLinear = model.NonLinear_both
        use_obs     = [] #CDL: self.observables contains the distribution for the observable used
        window_list = []
       
        if 'GC' in self.observables:
            Nbins_GC    = self.observables['GC']['Nbins']
            n_dict      = {i+1: self.observables['GC']['dist'][i](self.z) for i in range(0, Nbins_GC)}
            bias_binned = cosmo['bias'](self.observables['GC']['zmean'])

            window_list = window_list+[SplinedSourceWindow(source_type='counts', bias=bias_binned[i-1], z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_GC+1)]

            use_obs.append('GC')

        if 'WL' in self.observables:
            Nbins_WL  = self.observables['WL']['Nbins']
            n_dict    = {i+1: self.observables['WL']['dist'][i](self.z) for i in range(0, Nbins_WL)}
            '''CDL: The current implementation of CAMB doesn't include IA contribution into the WL kernels. So we exploit the fact that the IA 
                    window function has the same redshift dependence as the GCph one with the galaxy biased replaced by the IA amplitude. To do so 
                    CAMB expect a binned distribution for the IA contribution, that was evaluated fitting the commonly adopted IA model to a polynomial
                    form. For further details we refer you to Appendix B of https://arxiv.org/pdf/2512.19186'''
            IA_binned = cosmo['IA_term'](self.observables['WL']['zmean'])
            window_list = window_list+[SplinedSourceWindow(source_type='lensing', bias=1, z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_WL+1)]
            window_list = window_list+[SplinedSourceWindow(source_type='counts', bias=IA_binned[i-1], z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_WL+1)]

            use_obs.append('WL')
            

        if 'GWC' in self.observables:
            Nbins_GW  = self.observables['GWC']['Nbins']
            n_dict_gw    = {i+1: self.observables['GWC']['dist'][i](self.z) for i in range(0, Nbins_GW)}
            bias_binned_gw = cosmo['bias_GW'](self.observables['GWC']['zmean'])
            window_list = window_list+[SplinedSourceWindow(source_type='gwcounts', bias=bias_binned_gw[i-1], z=self.z, W=n_dict_gw[i]) 
                                       for i in range(1, Nbins_GW+1)]

            use_obs.append('GWC')
        
        if 'GWWL' in self.observables:
            Nbins_GW  = self.observables['GWWL']['Nbins']
            n_dict_gw    = {i+1: self.observables['GWWL']['dist'][i](self.z) for i in range(0, Nbins_GW)}
        
            window_list = window_list+[SplinedSourceWindow(source_type='gwamp', bias=1, z=self.z, W=n_dict_gw[i]) 
                                   for i in range(1, Nbins_GW+1)]

            use_obs.append('GWWL')
        pars.SourceWindows = window_list
        tini = time()
        results = camb.get_results(pars)

        cls = results.get_source_cls_dict(lmax=max(self.ells), raw_cl=True)

        if self.feedback:
            print('\033[1;32m' + f'CAMB took {time()-tini:.2f} s' + '\033[0m')

        return cls
   

    def set_camb_specs(self,pars):

        pars.Want_CMB = False
        pars.SourceTerms.limber_windows = True
        pars.SourceTerms.limber_phi_lmin = 2

        pars.SourceTerms.line_phot_dipole = False
        pars.SourceTerms.line_phot_quadrupole = False 
        pars.SourceTerms.line_basic = False
        pars.SourceTerms.line_distortions = False
        pars.SourceTerms.use_21cm_mK = False
        # counts = GC,  gwcounts = GWC, gwamp = GWWL, line = 21cm
        try:
            if 'GC' in self.obs_used:
                pars.SourceTerms.counts_density = True # Non-relativistic default
                pars.SourceTerms.counts_ISW = self.cases['ISW']
                pars.SourceTerms.counts_potential = self.cases['potential']
                pars.SourceTerms.counts_evolve = self.cases['evolve']
                pars.SourceTerms.counts_redshift = self.cases['redshift']
                pars.SourceTerms.counts_lensing =  self.cases['lensing']
                pars.SourceTerms.counts_velocity = self.cases['velocity']
                pars.SourceTerms.counts_radial = self.cases['velocity']
                pars.SourceTerms.counts_timedelay = self.cases['velocity']
            if 'GWC' in self.obs_used:
                pars.SourceTerms.gwcounts_density= True
                pars.SourceTerms.gwcounts_evolve= self.cases['evolve']
                pars.SourceTerms.gwcounts_gradpotential = self.cases['gradpotential']
                pars.SourceTerms.gwcounts_potential = self.cases['potential']
                pars.SourceTerms.gwcounts_ISW = self.cases['ISW']
                pars.SourceTerms.gwcounts_timedelay= self.cases['velocity']
                pars.SourceTerms.gwcounts_velocity= self.cases['velocity']
                pars.SourceTerms.gwcounts_lsd = self.cases['lsd']
                pars.SourceTerms.gwcounts_lensing = self.cases['lensing']
            if 'GWWL' in self.obs_used:
                pars.SourceTerms.gwlens_lensing = True
                pars.SourceTerms.gwlens_volume = self.cases['volume']
                pars.SourceTerms.gwlens_sw = self.cases['SW']
                pars.SourceTerms.gwlens_ISW = self.cases['ISW']
                pars.SourceTerms.gwlens_velocity = self.cases['velocity']
                pars.SourceTerms.gwlens_TD = self.cases['velocity']
   
            
        except Exception as e:
            sys.exit('\033[1;31m' + f"Error setting CAMB source terms. Valid cases: {', '.join(self.cases.keys())}. Error: {e}" + '\033[0m')
        return pars







    
        
    def get_cls(self,ells):
        
        cosmo=self.get_cosmo_dict(self.params)
        cls =self.get_source_Cls(cosmo)
        use_obs     = []
        #window_list = []
        '''CDL: Here we create our dataset in the format that we will use for the likelihood. 
        We create a dataframe with columns given by the different combinations of observables and rows given by the ells.
        The Cls are extracted from the CAMB output and interpolated at the ells of interest.'''
        Nbin={}
        if 'GC' in self.obs_used:
            GCcols = ['G{}xG{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(i,self.observables['GC']['Nbins']+1)]
            use_obs.append('GC')
            Nbin['GC']=self.observables['GC']['Nbins'] #10
        else: 
            Nbin['GC']=0

        if 'WL' in self.obs_used:
            WLcols = ['L{}xL{}'.format(i,j) for i in range(1,self.observables['WL']['Nbins']+1) for j in range(i,self.observables['WL']['Nbins']+1)]
            use_obs.append('WL')
            Nbin['WL'] = Nbin['GC']+self.observables['WL']['Nbins'] #10+10 = 20
            Nbin['IA'] = Nbin['WL']+self.observables['WL']['Nbins'] #10+2*20 = 30
        else :
            Nbin['WL'] = Nbin['GC']
            Nbin['IA'] = Nbin['WL']


        if 'GWC' in self.obs_used:
            GWCcols = ['WC{}xWC{}'.format(i,j) for i in range(1,self.observables['GWC']['Nbins']+1) for j in range(i,self.observables['GWC']['Nbins']+1)]
            use_obs.append('GWC')
            Nbin['GWC']=Nbin['IA']+self.observables['GWC']['Nbins'] #10+10+10+10 = 40
        else:
            Nbin['GWC']=Nbin['IA']

        if 'GWWL' in self.obs_used:
            GWWLcols = ['WL{}xWL{}'.format(i,j) for i in range(1,self.observables['GWWL']['Nbins']+1) for j in range(i,self.observables['GWWL']['Nbins']+1)]
            use_obs.append('GWWL')
            Nbin['GWL']=Nbin['GWC']+self.observables['GWWL']['Nbins'] #10+10+10+10+10 = 50
            
            


        
        if 'WL' in self.obs_used and 'GC' in self.obs_used:
            GGLcols = ['G{}xL{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['WL']['Nbins']+1)]
        if 'WL' in self.obs_used and 'GWC' in self.obs_used:
            LGWCcols = ['L{}xWC{}'.format(i,j) for i in range(1,self.observables['WL']['Nbins']+1) for j in range(1,self.observables['GWC']['Nbins']+1)]
        if 'GC' in  self.obs_used and 'GWC' in  self.obs_used:
            GGWCcols = ['G{}xWC{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['GWC']['Nbins']+1)]
        if 'GC' in  self.obs_used and 'GWWL' in  self.obs_used:
            GGWLcols = ['G{}xWL{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['GWWL']['Nbins']+1)]
        if 'WL' in self.obs_used and 'GWWL' in self.obs_used:
            LGWLcols = ['L{}xWL{}'.format(i,j) for i in range(1,self.observables['WL']['Nbins']+1) for j in range(1,self.observables['GWWL']['Nbins']+1)]
        if 'GWC' in self.obs_used and 'GWWL' in self.obs_used:
            GWCGWLcols = ['WC{}xWL{}'.format(i,j) for i in range(1,self.observables['GWC']['Nbins']+1) for j in range(1,self.observables['GWWL']['Nbins']+1)]
        
            

        all_cols=[]
        if 'WL' in self.obs_used: 
               all_cols = all_cols+WLcols
        if 'GC' in use_obs:
            if 'WL' in self.obs_used:
                all_cols = all_cols+GGLcols+GCcols
            else:
                all_cols = all_cols+GCcols
        if 'GWC' in self.obs_used:
            all_cols = all_cols+GWCcols
            if 'GC' in self.obs_used:
                all_cols = all_cols+GGWCcols
            if 'WL' in self.obs_used:
                all_cols = all_cols+LGWCcols
        if 'GWWL' in self.obs_used:
            all_cols = all_cols + GWWLcols
            if 'GC' in self.obs_used:
                all_cols = all_cols+GGWLcols
            if 'WL' in self.obs_used:
                all_cols = all_cols+LGWLcols
            if 'GWC' in self.obs_used:
                all_cols = all_cols+GWCGWLcols


        #######################################



        final_Cls = pd.DataFrame(columns=['ells']+all_cols)
        final_Cls['ells'] = ells
        
        tini = time()
        if 'GC' in self.obs_used:
            #Cls GC x GC

            gc_interp_dict = {'G{}xG{}'.format(bin1,bin2): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                    cls['W{}xW{}'.format(bin1, bin2)], kind='cubic')(ells)
                              for bin1 in range(1,Nbin['GC']+1) for bin2 in range(bin1,Nbin['GC']+1)}
            
            for key,val in gc_interp_dict.items():
                final_Cls[key] = val

            

        
        if 'WL' in self.obs_used:
            #Cls gamma x gamma
            gamma_interp_dict = {'L{}xL{}'.format(bin1-Nbin['GC'], bin2-Nbin['GC']): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                              cls['W{}xW{}'.format(bin1, bin2)], kind='cubic')(ells)
                                for bin1 in range(Nbin['GC']+1,Nbin['WL']+1) for bin2 in range(bin1,Nbin['WL']+1)}
            
            for key,val in gamma_interp_dict.items():
                final_Cls[key]=val

                
                
                
             #Cls IA x IA 
            
            IA_interp_dict = {'L{}xL{}'.format(bin1-Nbin['WL'], bin2-Nbin['WL']): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                           cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                                for bin1 in range(Nbin['WL']+1,Nbin['IA']+1) for bin2 in range(bin1,Nbin['IA']+1)}

            for key,val in IA_interp_dict.items():
                final_Cls[key]+=val
                

            #Cls gamma x IA
            gamma_IA_interp_dict = {'L{}xL{}'.format(bin1-Nbin['GC'], bin2-(Nbin['WL'])): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                   cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                                for bin1 in range (Nbin['GC']+1,Nbin['WL']+1) for bin2 in range(bin1+Nbin['GC'],Nbin['IA']+1)}

            for key,val in gamma_IA_interp_dict.items():
                final_Cls[key]+=val

            
        if 'GWC' in self.obs_used:
            #Cls GWC x GWC
            gwc_interp_dict = {'WC{}xWC{}'.format(bin1-(Nbin['IA']),bin2-(Nbin['IA'])): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                 cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                              for bin1 in range(Nbin['IA']+1,Nbin['GWC']+1) for bin2 in range(bin1,Nbin['GWC']+1)}
            
            for key,val in gwc_interp_dict.items():
                final_Cls[key] = val
                

        if 'GWWL' in self.obs_used:
            #Cls GWWL x GWWL

            gwl_interp_dict = {'WL{}xWL{}'.format(bin1-(Nbin['GWC']),bin2-(Nbin['GWC'])): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                   cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                              for bin1 in range(Nbin['GWC']+1,Nbin['GWL']+1) for bin2 in range(bin1,Nbin['GWL']+1)}
            
            for key,val in gwl_interp_dict.items():
                final_Cls[key] = val

        if 'WL' in self.obs_used and 'GC' in self.obs_used:
            #Cls gamma x GC
            gc_gamma_interp_dict = {}
            gamma_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']+1):
                for bin2 in range(Nbin['GC']+1,Nbin['WL']+1):
                    gc_gamma_interp_dict['G{}xL{}'.format(bin1, bin2-Nbin['GC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                             cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gamma_gc_interp_dict['L{}xG{}'.format(bin2-Nbin['GC'], bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                             cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

            
            for key,val in gc_gamma_interp_dict.items():
                final_Cls[key]= val
                

            for key,val in gamma_gc_interp_dict.items():
                final_Cls[key]= val


            
            #Cls Ia x GC
            gc_IA_interp_dict = {}
            IA_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']+1):
                for bin2 in range(Nbin['WL']+1,Nbin['IA']+1):
                    gc_IA_interp_dict['G{}xL{}'.format(bin1, bin2-(Nbin['WL']))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                            cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    IA_gc_interp_dict['L{}xG{}'.format(bin2-(Nbin['WL']), bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                            cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)
            
            
            for key,val in gc_IA_interp_dict.items():
                final_Cls[key]+= val

            for key,val in IA_gc_interp_dict.items():
                final_Cls[key]+= val

        if 'WL' in self.obs_used and 'GWC' in self.obs_used:
            #Cls gamma x GW
            gwc_gamma_interp_dict = {}
            gamma_gwc_interp_dict = {}
            for bin1 in range(Nbin['GC']+1,Nbin['WL']+1):
                for bin2 in range(Nbin['IA']+1,Nbin['GWC']+1):
                    gwc_gamma_interp_dict['L{}xWC{}'.format(bin1-Nbin['GC'], bin2-Nbin['IA'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                          cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gamma_gwc_interp_dict['WC{}xL{}'.format(bin2-Nbin['IA'], bin1-Nbin['GC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                                          cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)
            

            for key,val in gwc_gamma_interp_dict.items():
                final_Cls[key]= val

            for key,val in gamma_gwc_interp_dict.items():
                final_Cls[key]= val


            
            #Cls Ia x GWC
            gwc_IA_interp_dict = {}
            IA_gwc_interp_dict = {}
            for bin1 in range(Nbin['WL']+1,Nbin['IA']+1):
                for bin2 in range(Nbin['IA']+1,Nbin['GWC']+1):
                    gwc_IA_interp_dict['L{}xWC{}'.format(bin1-Nbin['WL'], bin2-Nbin['IA'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                       cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    IA_gwc_interp_dict['WC{}xL{}'.format(bin2-Nbin['IA'], bin1-Nbin['WL'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                                       cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

            for key,val in gwc_IA_interp_dict.items():
                final_Cls[key]+= val

            for key,val in IA_gwc_interp_dict.items():
                final_Cls[key]+= val

        if 'GWC' in self.obs_used and 'GC' in self.obs_used:
            #Cls GWC x GC
            gc_gwc_interp_dict = {}
            gwc_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']+1):
                
                for bin2 in range(Nbin['IA']+1,Nbin['GWC']+1):
                    
                    gc_gwc_interp_dict['G{}xWC{}'.format(bin1, bin2-Nbin['IA'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                            cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gwc_gc_interp_dict['WC{}xG{}'.format(bin2-Nbin['IA'], bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                            cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

              
            for key,val in gc_gwc_interp_dict.items():
                final_Cls[key]= val

            for key,val in gwc_gc_interp_dict.items():
                final_Cls[key]= val
        

        if 'GWWL' in self.obs_used and 'GC' in self.obs_used:
            #Cls GWWL x GC
            gc_gwl_interp_dict = {}
            gwl_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']+1):
                for bin2 in range(Nbin['GWC']+1,Nbin['GWL']+1):
                    gc_gwl_interp_dict['G{}xWL{}'.format(bin1, bin2-Nbin['GWC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                             cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gwl_gc_interp_dict['WL{}xG{}'.format(bin2-Nbin['GWC'], bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                             cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

              
            for key,val in gc_gwl_interp_dict.items():
                final_Cls[key]= val

            for key,val in gwl_gc_interp_dict.items():
                final_Cls[key]= val


        if 'WL' in self.obs_used and 'GWWL' in self.obs_used:
            #Cls gamma x GWWL
            gwl_gamma_interp_dict = {}
            gamma_gwl_interp_dict = {}
            for bin1 in range(Nbin['GC']+1,Nbin['WL']+1):
                for bin2 in range(Nbin['GWC']+1,Nbin['GWL']+1):
                    gwl_gamma_interp_dict['L{}xWL{}'.format(bin1-Nbin['GC'], bin2-Nbin['GWC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                           cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gamma_gwl_interp_dict['WL{}xL{}'.format(bin2-Nbin['GWC'], bin1-Nbin['GC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                                           cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)
            

            for key,val in gwl_gamma_interp_dict.items():
                final_Cls[key]= val

            for key,val in gamma_gwl_interp_dict.items():
                final_Cls[key]= val


            
            #Cls Ia x GWWL
            gwl_IA_interp_dict = {}
            IA_gwl_interp_dict = {}
            for bin1 in range(Nbin['WL']+1,Nbin['IA']+1):
                for bin2 in range(Nbin['GWC']+1,Nbin['GWL']+1):
                    gwl_IA_interp_dict['L{}xWL{}'.format(bin1-Nbin['WL'], bin2-Nbin['GWC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                        cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    IA_gwl_interp_dict['WL{}xL{}'.format(bin2-Nbin['GWC'], bin1-Nbin['WL'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                                        cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

            for key,val in gwl_IA_interp_dict.items():
                final_Cls[key]+= val

            for key,val in IA_gwl_interp_dict.items():
                final_Cls[key]+= val

        
        if 'GWWL' in self.obs_used and 'GWC' in self.obs_used:
            #Cls GWWL x GWC
            gwc_gwl_interp_dict = {}
            gwl_gwc_interp_dict = {}
            for bin1 in range(Nbin['IA']+1,Nbin['GWC']+1):
                for bin2 in range(Nbin['GWC']+1,Nbin['GWL']+1):
                    gwc_gwl_interp_dict['WC{}xWL{}'.format(bin1-Nbin['IA'], bin2-Nbin['GWC'])] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), 
                                                                                                          cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gwl_gwc_interp_dict['WL{}xWC{}'.format(bin2-Nbin['GWC'], bin1-Nbin['IA'],)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), 
                                                                                                           cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

              
            for key,val in gwc_gwl_interp_dict.items():
                final_Cls[key]= val

            for key,val in gwl_gwc_interp_dict.items():
                final_Cls[key]= val


        if self.feedback:
            print('\033[1;32m' + f'Cls packing took {time()-tini:.2f} s' + '\033[0m')
        return final_Cls
