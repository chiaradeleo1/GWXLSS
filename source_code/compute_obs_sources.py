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

    def __init__(self,params, extra, observables,ells,camb_path,case,source,feedback=False):

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
        self.source=source
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
        if self.source==True:
            self.Cls = self.get_cls( ells)
        else:
            self.Cls =self.get_cls_old()
        tend = time()
    
        if feedback:
            print('Cls computed in {:.1f} s'.format(tend - tini))




    

    def get_cosmo_dict(self,params): 
                
        nuis_keys  = ['_derived','a0','a1', 'a2', 'a3', 'a4', 'b0_poly', 'b1_poly', 'b2_poly', 'b3_poly', 'logA', 'omegam', 'omegab', 'sigma8']
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

        if self.source==False:
            cosmo=self.additional_cosmo_dict(cosmo_dict)  
            cosmo_dict=cosmo
        #print(cosmo_dict)
        return cosmo_dict


    def additional_cosmo_dict(self,cosmo_dict):
        sys.path.insert(0,self.camb_path)
        
        self.k_max_extrap= 1000000.0
        self.k_min_extrap= 0.00001
        import camb
        from   camb         import model
        
        cosmo_pars = cosmo_dict['cosmo_params']
        #print(cosmo_pars)
        self.z_camb=np.linspace(0.001,4.,100)
        if 'eft_params' in cosmo_dict:
            eft_params=cosmo_dict['eft_params']
            pars = camb.set_params(**cosmo_pars,**eft_params)
            

        else:
             pars = camb.set_params(**cosmo_pars)
            
        pars.NonLinear = model.NonLinear_both
        pars.set_matter_power(redshifts=self.z_camb, kmax=2.0)
        results = camb.get_results(pars)
        Omega0_cdm=results.get_Omega('cdm', z=0)
        Omega0_b=results.get_Omega('baryon', z=0)
        cosmo_dict['z']=self.z_camb
        cosmo_dict['Omm']= Omega0_cdm+Omega0_b
        cosmo_dict['comov_dist']=interp1d(self.z_camb,results.comoving_radial_distance(self.z_camb))
        cosmo_dict['angular_dist']=interp1d(self.z_camb,results.angular_diameter_distance(self.z_camb))
        cosmo_dict['H_Mpc'] = interp1d(self.z_camb, results.h_of_z(self.z_camb))

        cosmo_dict['Pk_linear'] = camb.get_matter_power_interpolator(pars, nonlinear=False, 
    hubble_units=False, k_hunit=False, kmax=50, var1='delta_tot',var2='delta_tot', zmax=self.z_camb[-1])
   


        cosmo_dict['Pk_delta']= camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=False, k_hunit=False, kmax=50, var1='delta_tot',var2='delta_tot', zmax=self.z_camb[-1])
        
        cosmo_dict['Pk_Weyl']=camb.get_matter_power_interpolator(pars, nonlinear=False, 
    hubble_units=False, k_hunit=False, kmax=50, var1='Weyl',var2='Weyl', zmax=self.z_camb[-1])
    
    



        ks = 0.001
        P_z_k = cosmo_dict['Pk_delta'].P(self.z_camb, ks)
        cosmo_dict['Dz'] = np.sqrt(P_z_k / cosmo_dict['Pk_delta'].P(0.001, ks))
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
        #print(pars)
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
            Nbins_GW  = self.observables['GW']['Nbins']
            pars.SourceTerms.gw_density=True
            pars.SourceTerms.gw_timedelay=False
            pars.SourceTerms.gw_evolve=False
            pars.SourceTerms.gw_velocity=False
            pars.SourceTerms.gw_ISW = False
            pars.SourceTerms.gw_lsd = False
            #(pars)
            n_dict_gw    = {i+1: self.observables['GW']['dist'][i](self.z) for i in range(0, Nbins_GW)}
        
            window_list = window_list+[SplinedSourceWindow(source_type='counts', bias=bias_binned[i-1], z=self.z, W=n_dict_gw[i]) 
                                       for i in range(1, Nbins_GW+1)]

            use_obs.append('GW')

        

        pars.SourceWindows = window_list

        tini = time()
        
        results = camb.get_results(pars)

        cls = results.get_source_cls_dict(lmax=max(self.ells), raw_cl=True)
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
            #print('GC',Nbins_GC, Nbin['GC'])

        if 'WL' in self.observables:
            Nbins_WL  = self.observables['WL']['Nbins']
            WLcols = ['L{}xL{}'.format(i,j) for i in range(1,Nbins_WL+1) for j in range(i,Nbins_WL+1)]
            use_obs.append('WL')
            if 'GC' in self.observables:
                Nbin['WL']=Nbins_GC+Nbins_WL+1
                Nbin['IA']= Nbins_GC+(2*Nbins_WL)+1
                #print('WL',Nbins_WL, Nbin['WL'])
                #print('IA',Nbin['IA'])

            else:
                Nbin['GC']=1
                Nbins_GC=0
                Nbin['WL']=Nbins_WL+1
                Nbin['IA']=(2*Nbins_WL)+1
##CDL to be improved to account for GCxGW only or WLxGW only
        if 'GW' in self.observables:
            Nbins_GW  = self.observables['GW']['Nbins']
            GWcols = ['W{}xW{}'.format(i,j) for i in range(1,Nbins_GW+1) for j in range(i,Nbins_GW+1)]
            use_obs.append('GW')
            if 'GC' in self.observables and 'WL' in self.observables:
                Nbin['GW']=Nbins_GC+(2*Nbins_WL)+Nbins_GW+1
                #print('GW',Nbins_GW, Nbin['GW'])

            else:
                Nbin['GW']=Nbins_GW+1

        
        if 'WL' in self.observables and 'GC' in self.observables:
            GGLcols = ['G{}xL{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['WL']['Nbins']+1)]
        if 'WL' in self.observables and 'GW' in self.observables:
            LGWcols = ['L{}xW{}'.format(i,j) for i in range(1,self.observables['WL']['Nbins']+1) for j in range(1,self.observables['GW']['Nbins']+1)]
        if 'GC' in  self.observables and 'GW' in  self.observables:
            GGWcols = ['G{}xW{}'.format(i,j) for i in range(1,self.observables['GC']['Nbins']+1) for j in range(1,self.observables['GW']['Nbins']+1)]
            

        all_cols=[]
        if 'WL' in use_obs: 
               all_cols = all_cols+WLcols
        if 'GC' in use_obs:
            if 'WL' in use_obs:
                all_cols = all_cols+GGLcols+GCcols
            else:
                all_cols = all_cols+GCcols
        if 'GW' in use_obs:
            all_cols = all_cols+GWcols
            if 'GC' in use_obs:
                all_cols = all_cols+GGWcols
            if 'WL' in use_obs:
                all_cols = all_cols+LGWcols

        #######################################



        final_Cls = pd.DataFrame(columns=['ells']+all_cols)
        final_Cls['ells'] = ells
        
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

            
        if 'GW' in use_obs:
            #Cls GW x GW

            gw_interp_dict = {'W{}xW{}'.format(bin1-(Nbins_GC+Nbins_WL+Nbins_WL),bin2-(Nbins_GC+Nbins_WL+Nbins_WL)): interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                              for bin1 in range(Nbin['IA'],Nbin['GW']) for bin2 in range(bin1,Nbin['GW'])}
            
            for key,val in gw_interp_dict.items():
                final_Cls[key] = val

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

        if 'WL' in use_obs and 'GW' in use_obs:
            #Cls gamma x GW
            gw_gamma_interp_dict = {}
            gamma_gw_interp_dict = {}
            for bin1 in range(Nbin['GC'],Nbin['WL']):
                for bin2 in range(Nbin['IA'],Nbin['GW']):
                    gw_gamma_interp_dict['L{}xW{}'.format(bin1-Nbins_GC, bin2-(Nbins_GC+(2*Nbins_WL)))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gamma_gw_interp_dict['W{}xL{}'.format(bin2-(Nbins_GC+(2*Nbins_WL)), bin1-Nbins_GC)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)
            

            for key,val in gw_gamma_interp_dict.items():
                final_Cls[key]= val

            for key,val in gamma_gw_interp_dict.items():
                final_Cls[key]= val


            
            #Cls Ia x GW
            gw_IA_interp_dict = {}
            IA_gw_interp_dict = {}
            for bin1 in range(Nbin['WL'],Nbin['IA']):
                for bin2 in range(Nbin['IA'],Nbin['GW']):
                    gw_IA_interp_dict['L{}xW{}'.format(bin1-(Nbins_GC+Nbins_WL), bin2-(Nbins_GC+(2*Nbins_WL)))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    IA_gw_interp_dict['W{}xL{}'.format(bin2-(Nbins_GC+(2*Nbins_WL)), bin1-(Nbins_GC+Nbins_WL))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

            for key,val in gw_IA_interp_dict.items():
                final_Cls[key]+= val

            for key,val in IA_gw_interp_dict.items():
                final_Cls[key]+= val

        if 'GW' in use_obs and 'GC' in use_obs:
            #Cls GW x GC
            gc_gw_interp_dict = {}
            gw_gc_interp_dict = {}
            for bin1 in range(1,Nbin['GC']):
                for bin2 in range(Nbin['IA'],Nbin['GW']):
                    gc_gw_interp_dict['G{}xW{}'.format(bin1, bin2-(Nbins_GC+(2*Nbins_WL)))] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin1, bin2)])), cls['W{}xW{}'.format(bin1, bin2)], kind='linear')(ells)
                    gw_gc_interp_dict['W{}xG{}'.format(bin2-(Nbins_GC+(2*Nbins_WL)), bin1)] = interp1d(np.arange(0, len(cls['W{}xW{}'.format(bin2, bin1)])), cls['W{}xW{}'.format(bin2, bin1)], kind='linear')(ells)

              
            for key,val in gc_gw_interp_dict.items():
                final_Cls[key]= val

            for key,val in gw_gc_interp_dict.items():
                final_Cls[key]= val


        if self.feedback:
            print('Cls packing took {:.2f} s'.format(time()-tini))
        return final_Cls






    

    def get_window(self):
        self.kernels = {}
        cosmo=self.get_cosmo_dict(self.params)
        #self.Nbins=self.distribution['GC']['Nbins']
        self.z            = np.linspace(0.001,4,500)
        
        
        # Computing kernels for different observations
        tini = time()
    
        if 'GC' in self.observables:
            self.Nbins_GC    = self.observables['GC']['Nbins']
            self.ni_GC     = self.observables['GC']['dist']
            self.kernels.update({'g'+str(i+1): self.gal_window(cosmo, i) for i in range(0, len(self.ni_GC))})
        
        if 'WL' in self.observables:
            self.Nbins_WL  = self.observables['WL']['Nbins']
            self.ni_WL    = self.observables['WL']['dist']
            self.leff = [np.array([self.lens_eff(cosmo, z, i) for z in cosmo['z']]) for i in range(len(self.ni_WL))]
            self.kernels.update({'d'+str(i+1): self.lens_window(cosmo, i) for i in range(0,len(self.ni_WL))})
            self.kernels.update({'i'+str(i+1): self.IA_window(cosmo, i) for i in range(0,len(self.ni_WL))})
        
        if 'GW' in self.observables:
            self.Nbins_GW    = self.observables['GW']['Nbins']
            self.ni_GW     = self.observables['GW']['dist']
            self.leff_gw = [np.array([self.gw_eff(cosmo, z, i) for z in cosmo['z']]) for i in range(len(self.ni_GW))]
            self.kernels.update({'gw'+str(i+1): self.gw_window(cosmo, i) for i in range(len(self.ni_GW))})
        tend = time()
        
        if self.feedback:
            print('Kernels computed in {:.1f} s'.format(tend - tini))

    
    def gal_window(self,cosmo,i):
        bias_binned = cosmo['bias'](self.observables['GC']['zmean'])
        Wgal = np.array([self.ni_GC[i](z)*bias_binned[i]*cosmo['H_Mpc'](z) for z in cosmo['z']])
        return Wgal

    def lens_window(self,cosmo,i):
        Wlens = np.array([cosmo['comov_dist'](z)*self.leff[i][ind] for ind,z in enumerate(cosmo['z'])])
        return Wlens

    def IA_window(self,cosmo,i):
        IA_binned = cosmo['IA_term'](self.observables['WL']['zmean'])
        WIA = np.array([self.ni_WL[i](z)*IA_binned[i]*cosmo['H_Mpc'](z) for z in cosmo['z']])
        return WIA

    def gw_window(self,cosmo,i):
        Wgw = np.array([cosmo['comov_dist'](z)*self.leff_gw[i][ind] for ind,z in enumerate(cosmo['z'])])
        return Wgw

    def lens_eff(self,cosmo,z,i):

        zp = np.linspace(z,cosmo['z'][-1],100)
        leff = trapz(self.ni_WL[i](zp)*(1-cosmo['comov_dist'](z)/cosmo['comov_dist'](zp)),x=zp)

        return leff
    def gw_eff(self,cosmo,z,i):

        zp = np.linspace(z,cosmo['z'][-1],100)
        leff_gw = trapz(self.ni_GW[i](zp)*(1-cosmo['comov_dist'](z)/cosmo['comov_dist'](zp)),x=zp)

        return leff_gw










    def get_Pell(self,cosmo,ell,z):
        
        kappa = (ell+0.5)/cosmo['comov_dist'](z)

        Pell = {'Pgg': cosmo['Pk_delta'].P(z,kappa),
                'Pgi': cosmo['Pk_delta'].P(z,kappa),
                'Pig': cosmo['Pk_delta'].P(z,kappa),
                'Pii': cosmo['Pk_delta'].P(z,kappa)}

        
        Pell.update({'Pdd': (cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pgwgw': (cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pdgw': (cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pgwd': (cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pdi': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pid': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pgwi': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pigw': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pdg': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pgd': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pggw': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa),
                         'Pgwg': np.sqrt(cosmo['Pk_Weyl'].P(z,kappa)/cosmo['Pk_linear'].P(z,kappa))*cosmo['Pk_delta'].P(z,kappa)})

        return Pell

    def get_cls_old(self):
        
        cosmo=self.get_cosmo_dict(self.params)
        integrand = np.array([1/(cosmo['H_Mpc'](z)*cosmo['comov_dist'](z)**2) for z in cosmo['z']]) #prefactor

        pspectra = np.array([[self.get_Pell(cosmo,ell,z) for z in cosmo['z']] for ell in self.ells]) #PS
        
        Cls = {}
        self.get_window()
        def mysplit(s):
            head = s.rstrip('0123456789')
            tail = s[len(head):]
            return head, tail

        zint = cosmo['z']#np.linspace(cosmo['z'][0],cosmo['z'][-1],10)
        #print(self.kernels.items())
        for n1,w1 in self.kernels.items():
            for n2,w2 in self.kernels.items():
                obs1,bin1 = mysplit(n1)
                obs2,bin2 = mysplit(n2)
                Cls.update({n1+'x'+n2: np.array([trapz([w1[zind]*w2[zind]*integrand[zind]*pspectra[ellind,zind]['P'+obs1+obs2] for zind,z in enumerate(zint)],x=zint) for ellind,ell in enumerate(self.ells)])}) #complete formula for Cls

        #Here build the final Cls in format ObsixObsj where Obs={GC=G,GW=W,WL=L} and i,j are the zbins -> e.g. L1xL1 means CLS evaluated using Lensing in the first bin
        if 'WL' in self.observables:
            WLcols = ['L{}xL{}'.format(i,j) for i in range(1,self.Nbins_WL+1) for j in range(i,self.Nbins_WL+1)]
        if 'GC' in self.observables:
            GCcols = ['G{}xG{}'.format(i,j) for i in range(1,self.Nbins_GC+1) for j in range(i,self.Nbins_GC+1)]
        if 'GW' in self.observables:
            GWcols = ['W{}xW{}'.format(i,j) for i in range(1,self.Nbins_GW+1) for j in range(i,self.Nbins_GW+1)]
        if 'GC' in self.observables and 'WL' in self.observables:    
            XCcols = ['G{}xL{}'.format(i,j) for i in range(1,self.Nbins_GC+1) for j in range(1,self.Nbins_WL+1)]
        if 'GC' in self.observables and 'GW' in self.observables:    
            GGWcols = ['G{}xW{}'.format(i,j) for i in range(1,self.Nbins_GC+1) for j in range(1,self.Nbins_GW+1)]
        if 'GW' in self.observables and 'WL' in self.observables:    
            LGWcols = ['W{}xL{}'.format(i,j) for i in range(1,self.Nbins_WL+1) for j in range(1,self.Nbins_GW+1)]
        
        if 'GC' in self.observables and 'WL' in self.observables and 'GW'  in self.observables:

            final_Cls = pd.DataFrame(columns=['ells']+WLcols+XCcols+GCcols+GWcols+GGWcols+LGWcols)
            
            
        if 'GC' not in self.observables and 'WL' in self.observables and 'GW' in self.observables:
          
            final_Cls = pd.DataFrame(columns=['ells']+WLcols+GWcols+LGWcols)

        if 'GC'  in self.observables and 'WL' not in self.observables and 'GW' in self.observables:
          
            final_Cls = pd.DataFrame(columns=['ells']+GCcols+GWcols+GGWcols)
            
            
        if 'GC' in self.observables and 'WL' in self.observables and 'GW' not in self.observables:

            final_Cls = pd.DataFrame(columns=['ells']+WLcols+GCcols+XCcols)
            
        
        if 'GC' not in self.observables and 'WL' not in self.observables and 'GW' in self.observables:
            
            final_Cls = pd.DataFrame(columns=['ells']+GWcols)
        final_Cls['ells'] = self.ells
        

########CHECK FOR DIFFERENT GC WL GW BINS

        if 'GC' in self.observables:
            for bin1 in range(1,self.Nbins_GC+1):
                for bin2 in range(bin1,self.Nbins_GC+1):
                    final_Cls['G{}xG{}'.format(bin1,bin2)] = Cls['g{}xg{}'.format(bin1,bin2)]
                    #print(final_Cls['G{}xG{}'.format(bin1,bin2)])

        if 'WL' in self.observables:
            for bin1 in range(1,self.Nbins_WL+1):
                for bin2 in range(bin1,self.Nbins_WL+1):
                    final_Cls['L{}xL{}'.format(bin1,bin2)] = Cls['d{}xd{}'.format(bin1,bin2)]+Cls['i{}xd{}'.format(bin1,bin2)]+Cls['i{}xi{}'.format(bin1,bin2)]
          

                    
                    
        if 'GW' in self.observables:
            for bin1 in range(1,self.Nbins_GW+1):
                for bin2 in range(bin1,self.Nbins_GW+1):
                    final_Cls['W{}xW{}'.format(bin1,bin2)] = Cls['gw{}xgw{}'.format(bin1,bin2)]
                    #print(final_Cls['W{}xW{}'.format(bin1,bin2)])

        
        if 'WL' in self.observables and 'GC' in self.observables:
            for bin1 in range(1,self.Nbins_GC+1):
                for bin2 in range(1,self.Nbins_WL+1):
                    final_Cls['G{}xL{}'.format(bin1,bin2)] = Cls['g{}xd{}'.format(bin1,bin2)]+Cls['g{}xi{}'.format(bin1,bin2)]
                    final_Cls['L{}xG{}'.format(bin1,bin2)] = Cls['d{}xg{}'.format(bin1,bin2)]+Cls['i{}xg{}'.format(bin1,bin2)]
                    
        if 'WL' in self.observables and 'GW' in self.observables:
            for bin1 in range(1,self.Nbins_WL+1):
                for bin2 in range(1,self.Nbins_GW+1):
                    final_Cls['L{}xW{}'.format(bin1,bin2)] = Cls['gw{}xd{}'.format(bin1,bin2)]+Cls['gw{}xi{}'.format(bin1,bin2)]
                    final_Cls['W{}xL{}'.format(bin1,bin2)] = Cls['d{}xgw{}'.format(bin1,bin2)]+Cls['i{}xgw{}'.format(bin1,bin2)]
                    
        if 'GC' in self.observables and 'GW' in self.observables:
            for bin1 in range(1,self.Nbins_GC+1):
                for bin2 in range(1,self.Nbins_GW+1):
                    final_Cls['G{}xW{}'.format(bin1,bin2)] = Cls['g{}xgw{}'.format(bin1,bin2)]
                    final_Cls['W{}xG{}'.format(bin1,bin2)] = Cls['gw{}xg{}'.format(bin1,bin2)]
            



        return final_Cls



    