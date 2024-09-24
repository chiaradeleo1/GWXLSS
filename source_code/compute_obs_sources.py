import sys
import numpy as np
import pandas as pd


from scipy.interpolate import interp1d
from scipy.integrate   import trapz

from copy      import deepcopy
from itertools import product
from time      import time

possible_observables = ['GC','WL','GW']

class get_obs:

    def __init__(self,params, extra, observables,ells,camb_path,case,feedback=False):

        #Reading used observables from source distribution
        #Checking that there is no weird stuff
        self.observables = observables
        self.params = deepcopy(params)
        if 'logA' in self.params:
            self.params['As'] = np.exp(self.params.pop('logA'))*1.e-10
        
        self.extra_params=extra
        self.case=case
        self.camb_path=camb_path
        self.ells=ells
        self.zinterps = np.logspace(-3,np.log10(5),500)
        self.k_max_Boltzmann = 10

        #MMmod: switch here if we want to customize
        self.params['dark_energy_model'] = 'ppf'
        
        for obs in self.observables.keys():
            if obs not in possible_observables:
                sys.exit('Unknown observable in source distribution file: {}'.format(obs))


        self.feedback     = feedback
        

        #MMmod: to be checked
        

        tini = time()
        self.Cls = self.get_cls( ells)
        tend = time()
    
        if feedback:
            print('Cls computed in {:.1f} s'.format(tend - tini))

    def get_cosmo_dict(self,params): 
                
        nuis_keys  = ['_derived','a0','a1', 'a2', 'a3', 'a4', 'b0_poly','b1_poly','b2_poly','b3_poly', 'logA', 'omegam', 'omegab', 'sigma8']
        nuis_keys = nuis_keys + list(self.extra_params.keys())
        
        cosmo_params = {par: params[par] for par in params.keys() if par not in nuis_keys}
        cosmo_dict = {'cosmo_params': cosmo_params}
        if 'EFTflag' in self.extra_params:
            flag=self.extra_params['EFTflag']
            if flag !=0:
                eftparams=self.extra_params
                cosmo_dict['eft_params'] = eftparams
            

        IA =[sum([params['a{}'.format(ind)]*np.power(z,ind) for ind in range(5)]) for z in self.zinterps] 
        cosmo_dict['IA_term']= interp1d(self.zinterps,IA)

        bgrid = [sum([params['b{}_poly'.format(ind)]*np.power(z,ind) for ind in range(4)]) for z in self.zinterps]
        cosmo_dict['bias'] = interp1d(self.zinterps,bgrid)
        
        return cosmo_dict

  
    def get_source_Cls(self,cosmo):
        
        sys.path.insert(0,self.camb_path)
        
        import camb
        from   camb.sources import GaussianSourceWindow, SplinedSourceWindow
        from   camb         import model, initialpower
        self.z            = np.linspace(0.001,4,500)
        cosmo_pars = cosmo['cosmo_params']
        
        #MMmod: removed print below to avoid cluttering
        #print(camb.__path__)
        if 'eft_params' in cosmo:
            eft_params=cosmo['eft_params']
            #print(eft_params)
            pars = camb.set_params(**cosmo_pars,**eft_params)
            #print(eft_params)

        else:
             pars = camb.set_params(**cosmo_pars)
        pars = self.set_camb_specs(pars,self.case)  #This sets the cases for CAMB (Limber & friends)
        pars.NonLinear = model.NonLinear_both
        use_obs     = []
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
            IA_binned = cosmo['IA_term'](self.observables['WL']['zmean'])

            window_list = window_list+[SplinedSourceWindow(source_type='lensing', bias=1, z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_WL+1)]
            window_list = window_list+[SplinedSourceWindow(source_type='counts', bias=IA_binned[i-1], z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_WL+1)]

            use_obs.append('WL')

        if 'GW' in self.observables:
            Nbins_GW    = self.observables['GW']['Nbins']
            n_dict      = {i+1: self.observables['GW']['dist'][i](self.z) for i in range(0, Nbins_GC)}
            bias_binned = cosmo['bias'](self.observables['GW']['zmean'])
            print(bias_binned)
            window_GW=[SplinedSourceWindow(source_type='gws', bias=bias_binned[i-1], z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_GW+1)]
            print(window_GW)
            window_list = window_list+[SplinedSourceWindow(source_type='gws', bias=bias_binned[i-1], z=self.z, W=n_dict[i]) 
                                       for i in range(1, Nbins_GW+1)]

        pars.SourceWindows = window_list

        tini = time()
        
        results = camb.get_results(pars)

        cls = results.get_source_cls_dict(lmax=max(self.ells), raw_cl=True)
        
        #print(cls.keys())
        if self.feedback:
            print('CAMB took {:.2f} s'.format(time()-tini))

        return cls


    def set_camb_specs(self,pars,case):#='simple'):
        pars.Want_CMB = False
        pars.Want_CMB = False
        pars.SourceTerms.limber_windows = True
        pars.SourceTerms.limber_phi_lmin = 2
        pars.SourceTerms.counts_ISW = True
        pars.SourceTerms.counts_potential = True
        pars.SourceTerms.counts_evolve = False
        pars.SourceTerms.line_phot_dipole = False
        pars.SourceTerms.line_phot_quadrupole = False 
        pars.SourceTerms.line_basic = False
        pars.SourceTerms.line_distortions = False
        pars.SourceTerms.use_21cm_mK = False
        if case == 'simple':
            pars.SourceTerms.counts_density = True
            pars.SourceTerms.counts_redshift = False
            pars.SourceTerms.counts_lensing = False
            pars.SourceTerms.counts_velocity = False
            
            pars.SourceTerms.counts_radial = False
            pars.SourceTerms.counts_timedelay = False
            
        elif case == 'total':
            
            pars.SourceTerms.counts_density = True
            pars.SourceTerms.counts_redshift = True
            pars.SourceTerms.counts_lensing = True
            pars.SourceTerms.counts_velocity = True
            
            pars.SourceTerms.counts_radial = True
            pars.SourceTerms.counts_timedelay = True
            
        elif case == 'density':
            
            pars.SourceTerms.counts_density = False
            pars.SourceTerms.counts_redshift = True
            pars.SourceTerms.counts_lensing = True
            pars.SourceTerms.counts_velocity = True
            
            pars.SourceTerms.counts_radial = True
            pars.SourceTerms.counts_timedelay = True
            
        elif case == 'redshift':
            
            #pars.SourceTerms.counts_density = False
            pars.SourceTerms.counts_redshift = False
            pars.SourceTerms.counts_lensing = True
            pars.SourceTerms.counts_velocity = True
            pars.SourceTerms.counts_radial = True
            pars.SourceTerms.counts_timedelay = True

        elif case == 'lensing':
            
            pars.SourceTerms.counts_density = True
            pars.SourceTerms.counts_redshift = True
            pars.SourceTerms.counts_lensing = False
            pars.SourceTerms.counts_velocity = True
            pars.SourceTerms.counts_radial = True
            pars.SourceTerms.counts_timedelay = True

        elif case == 'velocity':
            
            pars.SourceTerms.counts_density = True
            pars.SourceTerms.counts_redshift = True
            pars.SourceTerms.counts_lensing = True
            pars.SourceTerms.counts_velocity = False
            pars.SourceTerms.counts_radial = False
            pars.SourceTerms.counts_timedelay = False
            
        else:
            sys.exit('UNKNOWN CAMB CASE {}'.format(case))

        return pars
        
    def get_cls(self,ells):
        cosmo=self.get_cosmo_dict(self.params)
        cls=self.get_source_Cls(cosmo)
        use_obs     = []
        window_list = []
        Nbin={}
        if 'GC' in self.observables:
            Nbins_GC    = self.observables['GC']['Nbins']
            GCcols = ['G{}xG{}'.format(i,j) for i in range(1,Nbins_GC+1) for j in range(i,Nbins_GC+1)]
            use_obs.append('GC')
            Nbin['GC']=Nbins_GC+1
        if 'WL' in self.observables:
            Nbins_WL  = self.observables['WL']['Nbins']
            WLcols = ['L{}xL{}'.format(i,j) for i in range(1,Nbins_WL+1) for j in range(i,Nbins_WL+1)]
            use_obs.append('WL')
            if 'GC' in self.observables:
                Nbin['WL']=Nbins_GC+Nbins_WL+1
                Nbin['IA']= Nbins_GC+(2*Nbins_WL)+1

            else:
                Nbin['GC']=1
                Nbins_GC=0
                Nbin['WL']=Nbins_WL+1
                Nbin['IA']=(2*Nbins_WL)+1
        if 'WL' in self.observables and 'GC' in self.observables:
            GGLcols = ['G{}xL{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['WL']['Nbins']+1)]

        if 'GW' in self.observables:
            #sys.exit('GW not implemented yet!')
            use_obs.append('GW')

        
        if 'WL' in use_obs:
            if 'GC' in use_obs:
                all_cols = WLcols+GGLcols+GCcols
            else:
                all_cols = WLcols

        if 'GC' in use_obs and 'WL' not in use_obs:
            all_cols = GCcols

        #######################################



        final_Cls = pd.DataFrame(columns=['ells']+all_cols)
        final_Cls['ells'] = ells
        #Nbin={'GC': Nbins_GC+1,
         #     'WL': Nbins_GC+Nbins_WL+1,
          #    'IA': Nbins_GC+(2*Nbins_WL)+1}

        
        tini = time()
        if 'GC' in use_obs:
            #Cls GC x GC

            gc_interp_dict = {'G{}xG{}'.format(bin1,bin2): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                              for bin1 in range(1,Nbin['GC']) for bin2 in range(bin1,Nbin['GC'])}
            
            for key,val in gc_interp_dict.items():
                final_Cls[key] = val


            

        
        if 'WL' in use_obs:
            #Cls gamma x gamma
            gamma_interp_dict = {'L{}xL{}'.format(bin1-Nbins_GC, bin2-Nbins_GC): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                                for bin1 in range(Nbin['GC'],Nbin['WL']) for bin2 in range(bin1,Nbin['WL'])}
            
                    
            for key,val in gamma_interp_dict.items():
                final_Cls[key]=val
                
                
             #Cls IA x IA 
            
            IA_interp_dict = {'L{}xL{}'.format(bin1-(Nbins_GC+Nbins_WL), bin2-(Nbins_GC+Nbins_WL)): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                                for bin1 in range(Nbin['WL'],Nbin['IA']) for bin2 in range(bin1,Nbin['IA'])}

            for key,val in IA_interp_dict.items():
                final_Cls[key]+=val

            
            #Cls gamma x IA
            gamma_IA_interp_dict = {'L{}xL{}'.format(bin1-Nbins_GC, bin2-(Nbins_GC+Nbins_WL)): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                                for bin1 in range (Nbin['GC'],Nbin['WL']) for bin2 in range(bin1+Nbins_WL,Nbin['IA'])}

            for key,val in gamma_IA_interp_dict.items():
                final_Cls[key]+=val

            
        
        if 'WL' in use_obs and 'GC' in use_obs:
            #Cls gamma x GC
            gc_gamma_interp_dict = {}
            gamma_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']):
                for bin2 in range(Nbin['GC'],Nbin['WL']):
                    gc_gamma_interp_dict['G{}xL{}'.format(bin1, bin2-Nbins_GC)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gamma_gc_interp_dict['L{}xG{}'.format(bin2-Nbins_GC, bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)


            for key,val in gc_gamma_interp_dict.items():
                final_Cls[key]= val

            for key,val in gamma_gc_interp_dict.items():
                final_Cls[key]= val


            
            #Cls Ia x GC
            gc_IA_interp_dict = {}
            IA_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']):
                for bin2 in range(Nbin['WL'],Nbin['IA']):
                    gc_IA_interp_dict['G{}xL{}'.format(bin1, bin2-(Nbins_GC+Nbins_WL))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    IA_gc_interp_dict['L{}xG{}'.format(bin2-(Nbins_GC+Nbins_WL), bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

            for key,val in gc_IA_interp_dict.items():
                final_Cls[key]+= val

            for key,val in IA_gc_interp_dict.items():
                final_Cls[key]+= val
                    


        if self.feedback:
            print('Cls packing took {:.2f} s'.format(time()-tini))
        return final_Cls

   

