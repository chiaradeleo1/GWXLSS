import sys,re
import numpy  as np
import pandas as pd

from source_code.compute_obs_sources import get_obs
from source_code.covariance_utils    import covariance_einsum

from time      import time
from copy      import deepcopy
from itertools import product



class get_Fisher:

    def __init__(self,fiducial,free_params,ells,deltas,obs,obs_settings,galaxy_specs,GW_specs,dertype):

        self.ells         = ells
        self.observables  = obs
        self.galaxy_specs = galaxy_specs
        self.GW_specs     = GW_specs
        self.fiducial     = fiducial
        self.deltas       = deltas
        self.obs_settings = obs_settings
        self.free_params  = free_params
        self.param_names  = list(free_params.keys())
        self.param_values = [obs_settings['extra'][name] if name in self.obs_settings['extra'] else fiducial[name] for name in self.param_names]
        self.param_deltas = [free_params[name]['variation'] for name in self.param_names]
        self.dertype      = dertype
        

        #MMmod: this renaming is not really useful, but I wanted to preserve what is in compute obs
        self.renamed_obs = []
        for obs in self.observables:
            if obs == 'GC':
                self.renamed_obs.append('G')
            elif obs == 'WL':
                self.renamed_obs.append('L')
            elif obs == 'GWWL':
                self.renamed_obs.append('WL')
            elif obs == 'GWC':
                self.renamed_obs.append('WC')

        self.Nbins   = {new_obs: self.observables[obs]['Nbins'] for new_obs,obs in zip(self.renamed_obs,self.observables)}
        self.maxbins = max(list(self.Nbins.values()))
        self.cols    = [f'{o}{ind+1}' for o in self.renamed_obs for ind in range(self.maxbins)]

        print('')
        print('Computing fiducial observables...')
        tini = time()
        self.fidobs = get_obs(fiducial,self.observables,self.ells,self.obs_settings,feedback=False).Cls
        #MMmod: this renaming is not really useful, but I wanted to preserve what is in compute obs
        print('...done in {:.2f} s'.format(time()-tini))



    def numerical_derivative(self):
        #MMmod: more reliable method to be added here!!! E.g. STEM

        ells = self.fidobs['ells']
        derivs   = {param: {col: np.zeros(len(self.ells)) for col in self.fidobs.columns}
                    for param in self.param_names }

        for i, (param, delta) in enumerate(zip(self.param_names, self.param_deltas)):
            print('...computing derivative for {}...'.format(param))
            fiducial_plus  = deepcopy(self.fiducial)
            fiducial_minus = deepcopy(self.fiducial)
            
            
            if param in self.fiducial:
                if self.fiducial[param] == 0:
                    epsilon = delta
                else:
                    epsilon = abs(self.fiducial[param])*delta
                
                fiducial_plus[param]  += epsilon
                fiducial_minus[param] -= epsilon
            
            if self.dertype == '2PT':
                Cls_plus  = get_obs(fiducial_plus, self.observables, self.ells, self.obs_settings, feedback=False).Cls
                Cls_minus = get_obs(fiducial_minus, self.observables, self.ells, self.obs_settings, feedback=False).Cls

                derivs[param] = (Cls_plus-Cls_minus)/(2*epsilon)
            elif self.dertype == 'polynomial':
                Nevals      = 5
                eval_points = np.linspace(fiducial_minus[param],fiducial_plus[param],Nevals)
                eval_obs    = []
                for val in eval_points:
                    locpars = deepcopy(self.fiducial)
                    locpars[param] = val

                    eval_obs.append(get_obs(locpars, self.observables, self.ells, self.obs_settings, feedback=False).Cls)

                for col in eval_obs[0].columns:
                    for ind,ell in enumerate(ells):
                        fit = np.polyfit(eval_points,[eval_obs[i].at[ind,col] for i in range(Nevals)],4)
                        pol = np.poly1d(fit)
                        der = np.polyder(pol)#why not callable???
                        derivs[param][col][ind] = der(self.fiducial[param])



            derivs[param]['ells'] = ells
            derivs[param] = pd.DataFrame.from_dict(derivs[param])

        return derivs



    def compute_covmat(self):
        
        ells = self.fidobs['ells'].values
        Nell = {k: [0.]*len(ells) for k in self.fidobs.columns}

        for i in range(1,self.maxbins+1):
            for obs in self.renamed_obs:
                if obs == 'G':
                    ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.Nbins[obs])*3600*(180/np.pi)**2
                    Nell['{}{}x{}{}'.format(obs,i,obs,i)] = [(1/ngalbin)]*len(ells)
                elif obs == 'L':
                    ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.Nbins[obs])*3600*(180/np.pi)**2
                    Nell['{}{}x{}{}'.format(obs,i,obs,i)] = [(self.galaxy_specs['sigma_eps']**2/(2*ngalbin))]*len(ells)
                elif obs == 'WL':
                    self.ngwcbin = self.GW_specs['N_gw']/self.Nbins[obs]
                    Nell['{}{}x{}{}'.format(obs,i,obs,i)] = [(self.GW_specs['sigma_eps_gw']**2/self.ngwcbin)]*len(ells)
                elif obs == 'WC':
                    self.ngwcbin = self.GW_specs['N_gw']/self.Nbins[obs]
                    Nell['{}{}x{}{}'.format(obs,i,obs,i)] = [(1/self.ngwcbin)]*len(ells)

        self.noise = pd.DataFrame.from_dict({'ells': ells}|Nell)

        #print(self.noise.columns)
        #print(self.fidobs.columns)

        #MMmod: can we account for different fsky in different probes??
        #probably, see original cosmicfish approach
        #needs to add fsky after the covmat calculation and use this with fsky=1
        fsky      = self.galaxy_specs['fsky']
        ###############################################################


        Delta_ell = self.deltas
        Nobs      = len(self.observables)

        err_for_cov = np.zeros((Nobs,Nobs,len(ells),self.maxbins,self.maxbins))
        cls_for_cov = np.zeros((Nobs,Nobs,len(ells),self.maxbins,self.maxbins))

        for o1,obs1 in enumerate(self.renamed_obs):
            for o2,obs2 in enumerate(self.renamed_obs):
                for i in range(self.maxbins):
                    for j in range(self.maxbins):
                        if obs1 == obs2 and j<i:
                            self.noise[obs1+str(i+1)+'x'+obs2+str(j+1)]  = self.noise[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)] = self.fidobs[obs1+str(j+1)+'x'+obs2+str(i+1)]
                        for ell_ind,ell in enumerate(ells):
                            err_for_cov[o1,o2,ell_ind,i,j] = self.noise.at[ell_ind,obs1+str(i+1)+'x'+obs2+str(j+1)]
                            cls_for_cov[o1,o2,ell_ind,i,j] = self.fidobs.at[ell_ind,obs1+str(i+1)+'x'+obs2+str(j+1)]
        

        covmat = covariance_einsum(cls_for_cov,err_for_cov,fsky,ells,Delta_ell,return_only_diagonal_ells=True)


        return covmat

    def split_num(self,s):
        head = s.rstrip('0123456789')
        tail = s[len(head):]
        return head, tail


    def fisher_matrix(self):
        
        Fisher = np.zeros((len(self.ells), len(self.param_names), len(self.param_names)))
        print('')
        print('Computing covariance...')
        tini = time()
        covmat = self.compute_covmat()
        print('..done in {:.2f} s'.format(time()-tini))
        print('')
        print('Computing derivatives...')
        tini = time()
        derivs = self.numerical_derivative()
        print('...done in {:.2f} s'.format(time()-tini))

        #Column ordering for the matrix
        cols = {}
        for o1,obs1 in enumerate(self.renamed_obs):
            cols[obs1] = [obs1+'{}x'.format(i)+obs1+'{}'.format(j) for i in range(1,self.maxbins+1) for j in range(i,self.maxbins+1)] 
            for o2,obs2 in enumerate(self.renamed_obs):
                if o2>o1:
                    cols[obs1+'x'+obs2] = [obs1+'{}x'.format(i)+obs2+'{}'.format(j) for i in range(1,self.maxbins+1) for j in range(1,self.maxbins+1)]

        all_cols = []

        for obscomb,columns in cols.items():
            all_cols = all_cols + columns

        str_to_ind = {obs: ind for ind,obs in enumerate(self.renamed_obs)}

        for ellind,ell in enumerate(self.fidobs['ells']):
    
            packed_covmat = pd.DataFrame(columns=all_cols,index=all_cols,dtype='float')
    
            for ind1,col in enumerate(all_cols):
                bin1,bin2 = re.split('x',col)
                oi1,i1 = self.split_num(bin1)
                oj1,j1 = self.split_num(bin2)
    
                for ind2,row in enumerate(all_cols):
                    bin1,bin2 = re.split('x',row)
                    oi2,i2 = self.split_num(bin1)
                    oj2,j2 = self.split_num(bin2)
            
                    packed_covmat.at[row,col] = covmat[str_to_ind[oi1],str_to_ind[oj1],str_to_ind[oi2],str_to_ind[oj2],
                                                       ellind,int(i1)-1,int(j1)-1,int(i2)-1,int(j2)-1]
            
                #packed_covmat.index = packed_covmat.columns

            invcov = np.linalg.inv(packed_covmat)

            for i1,par1 in enumerate(self.free_params):
                for i2,par2 in enumerate(self.free_params):

                    der1 = np.array([derivs[par1].at[ellind,col] for col in all_cols])
                    der2 = np.array([derivs[par2].at[ellind,col] for col in all_cols])

                    Fisher[ellind,i1,i2] = np.dot(der1,np.dot(invcov,np.transpose(der2)))

        Fisher_final = np.sum(Fisher, axis=0)

        
        return Fisher_final

    
