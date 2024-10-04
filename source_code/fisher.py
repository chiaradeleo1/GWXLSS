import numpy as np
import copy
from copy      import deepcopy
from source_code.compute_obs_sources import get_obs
import pandas            as pd
from itertools import product



class get_Fisher:
    def __init__(self, fiducial, free_params, ells, deltas, obs, obs_settings, galaxy_specs,covmat_type):

        self.ells          = ells
        self.observables   = obs
        self.galaxy_specs  = galaxy_specs
        self.fiducial      = fiducial
        self.deltas        = deltas
        self.obs_settings  = obs_settings
        self.free_params   = free_params
        self.param_names   = list(free_params.keys())
        self. param_values = [obs_settings['extra'][name] if name in self.obs_settings['extra'] else fiducial[name] for name in self.param_names]
        self.param_deltas  = [free_params[name]['variation'] for name in self.param_names]
        self.covmat_type   = covmat_type
        self.obs           = []

        
        if 'GC' in self.observables:
            self.Nbins_gc=self.observables['GC']['Nbins']
            self.obs.append('G')
            
        if 'WL' in self.observables:
            self.Nbins_wl=self.observables['WL']['Nbins']
            self.obs.append('L')

        
        if 'GW' in self.observables:
            self.Nbins_gw=self.observables['GW']['Nbins']
            self.obs.append('W')
            
        self.maxbins=max(self.Nbins_gc, self.Nbins_wl, self.Nbins_gw)
        ######################################################
        self.cols = [f'{o}{ind+1}' for o in self.obs for ind in range(self.maxbins)]

        print('')
        print('Computing fiducial observables...')
        tini = time()
        self.fidobs = get_obs(fiducial,self.obs_settings['extra'],self.observables,self.ells,self.obs_settings['camb_path'],self.obs_settings['case'],feedback=False).Cls
        print('...done in {:.2f} s'.format(time()-tini))



    def numerical_derivative(self):
        n_params = len(self.param_names)
        n_ells = len(self.ells)
        derivs = {param: {f"{ind1}x{ind2}": np.zeros(len(self.ells)) for ind1 in self.cols for ind2 in self.cols}
            for param in self.param_names }
        
        for i, (param, delta) in enumerate(zip(self.param_names, self.param_deltas)):
            fiducial_plus = copy.deepcopy(self.fiducial)
            fiducial_minus = copy.deepcopy(self.fiducial)
            extra_plus = copy.deepcopy(self.obs_settings['extra'])
            extra_minus = copy.deepcopy(self.obs_settings['extra'])
            
            epsilon = self.fiducial[param] * delta
            fiducial_plus[param] += epsilon
            fiducial_minus[param] -= epsilon
            
            Cls_plus = get_obs(fiducial_plus, self.obs_settings['extra'], self.observables, self.ells, self.obs_settings['camb_path'], self.obs_settings['case'], feedback=False).Cls
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





    


    def compute_covmat(self):
        
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







    def fisher_matrix(self):
        
        Fisher = np.zeros((len(self.ells), len(self.param_names), len(self.param_names)))
        derivs = self.numerical_derivative()
        covmat= self.compute_covmat()
        
    
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
        
        return Fisher_final

    