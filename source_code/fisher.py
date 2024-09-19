import sys
import numpy  as np
import pandas as pd

from source_code.compute_obs_sources import get_obs
from source_code.covariance_utils    import covariance_einsum

from time      import time
from copy      import deepcopy
from itertools import product



class get_Fisher:

    def __init__(self,fiducial,free_params,ells,deltas,obs,obs_settings,galaxy_specs,covmat_type):

        self.ells          = ells
        self.observables   = obs
        self.galaxy_specs  = galaxy_specs
        self.fiducial      = fiducial
        self.deltas        = deltas
        self.obs_settings  = obs_settings
        self.free_params   = free_params
        self.param_names   = list(free_params.keys())
        self. param_values = [fiducial[name] for name in self.param_names]
        self.param_deltas  = [free_params[name]['variation'] for name in self.param_names]
        self.covmat_type   = covmat_type
        self.obs           = []

        #MMmod: this assumes only GC and WL!!! To be extended!
        if 'GC' in self.observables:
            self.Nbins_gc=self.observables['GC']['Nbins']
            self.obs.append('G')
            
        if 'WL' in self.observables:
            self.Nbins_wl=self.observables['WL']['Nbins']
            self.obs.append('L')
            
        self.maxbins=max(self.Nbins_gc, self.Nbins_wl)
        ######################################################
        self.cols = [f'{o}{ind+1}' for o in self.obs for ind in range(self.maxbins)]

        print('')
        print('Computing fiducial observables...')
        tini = time()
        self.fidobs = get_obs(fiducial,self.obs_settings['extra'],self.observables,self.ells,self.obs_settings['camb_path'],self.obs_settings['case'],feedback=False).Cls
        print('...done in {:.2f} s'.format(time()-tini))


    def numerical_derivative(self):
        #MMmod: more reliable method to be added here!!! E.g. STEM
        n_params = len(self.param_names)
        n_ells   = len(self.ells)
        derivs   = {param: {f"{ind1}x{ind2}": np.zeros(len(self.ells)) for ind1 in self.cols for ind2 in self.cols}
                    for param in self.param_names }
        
        for i, (param, delta) in enumerate(zip(self.param_names, self.param_deltas)):
            print('...computing derivative for {}...'.format(param))
            fiducial_plus  = deepcopy(self.fiducial)
            fiducial_minus = deepcopy(self.fiducial)
            
            #MM: added to avoid 0 value epsilons if fiducial is 0
            if self.fiducial[param] == 0:
                epsilon = delta
            else:
                epsilon = abs(self.fiducial[param])*delta
            
            fiducial_plus[param]  += epsilon
            fiducial_minus[param] -= epsilon
            
            Cls_plus  = get_obs(fiducial_plus, self.obs_settings['extra'], self.observables, self.ells, self.obs_settings['camb_path'], self.obs_settings['case'], feedback=False).Cls
            Cls_minus = get_obs(fiducial_minus, self.obs_settings['extra'], self.observables, self.ells, self.obs_settings['camb_path'], self.obs_settings['case'], feedback=False).Cls
            for ia, aa in enumerate(self.cols):
                for ib, bb in enumerate(self.cols):
                    if ia>ib:
                        #reverse_key = f"{bb}x{aa}"
                        derivs[param][aa+'x'+bb]=derivs[param][bb+'x'+aa]
                   
                    else:
                        #print(Cls_plus[aa+'x'+bb])
                        deriv_value = (Cls_plus[aa+'x'+bb] - Cls_minus[aa+'x'+bb]) / (2 * epsilon)
                        derivs[param][aa+'x'+bb] = deriv_value
    
        return derivs


    def get_cls_noisy(self):
        ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.numbins)*3600*(180/np.pi)**2
        calc_obs= get_obs(self.fiducial,self.obs_settings['extra'], self.observables, self.ells, self.obs_settings['camb_path'], self.obs_settings['case'], feedback=False)
        eps_error = self.galaxy_specs['sigma_eps']
        noisy_cls =  copy.deepcopy(calc_obs.Cls)
        #print(calc_obs.Cls)
        
    
        for ind in (1,self.numbins):
            for obs in self.obs:
                if obs == 'GC':
                    noisy_cls['G'+str(ind)+'xG'+str(ind)] += 1/ngalbin
                if obs == 'WL':
                    noisy_cls['L'+str(ind)+'xL'+str(ind)] += (eps_error**2)/(2*ngalbin )
        return noisy_cls



    def compute_covmat_second_order(self):
        print('Computing covmat')
        
        noisy_cls=self.get_cls_noisy()
        covmat = []
        
        binrange = range(1, self.numbins+1 )
        
        for ind, ell in enumerate(noisy_cls['ells']):
            covdf = pd.DataFrame(index=self.cols, columns=self.cols).fillna(0.)
            for obs1, obs2, bin1, bin2 in product(self.obs, self.obs, binrange, binrange):
                #print(obs1,bin1,obs2,bin2)
                if f'{obs1}{bin1}x{obs2}{bin2}' in noisy_cls:
                    
                    covdf.at[f'{obs1}{bin1}', f'{obs2}{bin2}'] = noisy_cls[f'{obs1}{bin1}x{obs2}{bin2}'][ind] / np.sqrt(self.galaxy_specs['fsky'])
                    
                else:
                    #print('key', f'{obs1}{bin1}x{obs2}{bin2}', 'not found in noisy cls' )
                    covdf.at[f'{obs1}{bin1}', f'{obs2}{bin2}'] = covdf.at[f'{obs2}{bin2}', f'{obs1}{bin1}']
                    #print('hello',covdf)
            covmat.append(covdf)
            
        return covmat


    def compute_covmat_fourth_order(self):

        ells = self.fidobs['ells'].values
        Nell = {k: [0.]*len(ells) for k in self.fidobs.columns}

        for i in range(1,self.maxbins+1):
            if 'GC' in self.observables and i<=self.Nbins_gc:
                ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.Nbins_gc)*3600*(180/np.pi)**2
                Nell['G{}xG{}'.format(i,i)] = [(1/ngalbin)]*len(ells)
        #print(Nell['G{}xG{}'.format(i,i)])
            if 'WL' in self.observables and i<= self.Nbins_wl:
                ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.Nbins_wl)*3600*(180/np.pi)**2
                Nell['L{}xL{}'.format(i,i)] = [(self.galaxy_specs['sigma_eps']**2/ngalbin)]*len(ells)

        #MMmod: can we account for different fsky in different probes??
        fsky      = self.galaxy_specs['fsky']
        Delta_ell = self.deltas


        #MMmod: warning! From here assumes only GC/WL. TOBECHANGED!!!
        err_for_cov = np.zeros((2,2,len(ells),self.Nbins_wl,self.Nbins_wl))
        cls_for_cov = np.zeros((2,2,len(ells),self.Nbins_wl,self.Nbins_wl))

        obs_list = []
        if 'GC' in self.observables:
            obs_list.append('G')
        if 'WL' in self.observables:
            obs_list.append('L')

        for o1,obs1 in enumerate(obs_list):
            if o1==0:
                Nbins1=self.Nbins_gc
            elif o1==1:
                Nbins1=self.Nbins_wl
                for o2,obs2 in enumerate(obs_list):
                    if o2==0:
                        Nbins2=self.Nbins_gc
                    elif o2==1:
                        Nbins2=self.Nbins_wl
                    for i in range(Nbins1):
                        for j in range(Nbins2):
                            for ell_ind,ell in enumerate(ells):
                                if obs1 == obs2 and j<i:
                                    Nell[obs1+str(i+1)+'x'+obs2+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                                    self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)] = self.fidobs[obs1+str(j+1)+'x'+obs2+str(i+1)]
                        #print(obs1+str(i+1)+'x'+obs2+str(j+1))

                                err_for_cov[o1,o2,ell_ind,i,j] = Nell[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
                                cls_for_cov[o1,o2,ell_ind,i,j] = self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]


        covmat = covariance_einsum(cls_for_cov,err_for_cov,fsky,ells,Delta_ell,return_only_diagonal_ells=True)


        return covmat


    def fisher_matrix(self):
        
        Fisher = np.zeros((len(self.ells), len(self.param_names), len(self.param_names)))
        print('')
        print('Computing covariance...')
        tini = time()
        if self.covmat_type == 0:
            covmat = self.compute_covmat_second_order()
        elif self.covmat_type == 1:
            covmat = self.compute_covmat_fourth_order()
        else:
            sys.exit('Unknown covmat type: {}'.format(self.covmat_type))
        print('..done in {:.2f} s'.format(time()-tini))
        print('')
        print('Computing derivatives...')
        tini = time()
        derivs = self.numerical_derivative()
        print('...done in {:.2f} s'.format(time()-tini))
        
   
        if self.covmat_type == 0:
            covarr = np.zeros(((len(self.ells)), len(self.cols), len(self.cols)))
            der1 = np.zeros((len(self.cols), len(self.cols)))
            der2 = np.zeros((len(self.cols), len(self.cols)))
    
    
            for i_ell in range(len(self.ells)):
                covarr[i_ell, :,:] = covmat[i_ell]
                covdf = covarr[i_ell, :, :]
                inv_covmat = np.linalg.pinv(covdf)
                for ind1,par1 in enumerate(self.free_params):
                    for ind2,par2 in enumerate(self.free_params):
                        if ind1>ind2: #2>1 la compoente Fisher 1,2 simmetrica a 2,1
                            Fisher[i_ell, ind1,ind2] = Fisher[i_ell, ind2,ind1]
                            continue
    
                        else:
                            for ia, aa in enumerate(self.cols):
                                    for ib, bb in enumerate(self.cols):
                                        der1[ia, ib] = derivs[par1][aa+'x'+bb][i_ell]
                                        der2[ia, ib] = derivs[par2][aa+'x'+bb][i_ell]
                                    
                            mat1   = der1.dot(inv_covmat)
                            mat2   = inv_covmat.dot(mat1)
                            mat3   = der2.dot(mat2)
                            trace  = np.trace(mat3)
                        
                            Fisher[i_ell, ind1, ind2] = (trace*(self.ells[i_ell]+0.5)*self.deltas[i_ell])
            Fisher_final = np.sum(Fisher, axis=0)
        elif self.covmat_type == 1:
            sys.exit('Not ready yet!')
        
        return Fisher_final

    
