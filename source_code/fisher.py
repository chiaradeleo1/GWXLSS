import sys,re
import numpy  as np
import pandas as pd

from source_code.compute_obs_sources import get_obs
from source_code.covariance_utils    import covariance_einsum

from time      import time
import copy
from copy      import deepcopy
from itertools import product



class get_Fisher:

    def __init__(self,fiducial,free_params,ells,deltas,obs,obs_settings,galaxy_specs,GW_specs,covmat_type):

        self.ells          = ells
        self.observables   = obs
        self.galaxy_specs  = galaxy_specs
        self.GW_specs      = GW_specs
        self.fiducial      = fiducial
        self.deltas        = deltas
        self.obs_settings  = obs_settings
        self.free_params   = free_params
        self.param_names   = list(free_params.keys())
        self. param_values = [obs_settings['extra'][name] if name in self.obs_settings['extra'] else fiducial[name] for name in self.param_names]
        self.param_deltas  = [free_params[name]['variation'] for name in self.param_names]
        self.covmat_type   = covmat_type
        self.obs           = []
        
        self.maxbins=0
        if 'GC' in self.observables:
            self.Nbins_gc=self.observables['GC']['Nbins']
            self.obs.append('G')
            self.maxbins=max(self.Nbins_gc, self.maxbins)
            
        if 'WL' in self.observables:
            self.Nbins_wl=self.observables['WL']['Nbins']
            self.obs.append('L')
            self.maxbins=max(self.Nbins_wl, self.maxbins)

        
        if 'GWWL' in self.observables:
            self.Nbins_gwl=self.observables['GWWL']['Nbins']
            self.obs.append('WL')
            self.maxbins=max(self.Nbins_gwl, self.maxbins)
        
        if 'GWC' in self.observables:
            self.Nbins_gwc=self.observables['GWC']['Nbins']
            self.obs.append('WC')
            self.maxbins=max(self.Nbins_gwc, self.maxbins)
            
        ######################################################
        self.cols = [f'{o}{ind+1}' for o in self.obs for ind in range(self.maxbins)]

        print('')
        print('Computing fiducial observables...')
        tini = time()
        self.fidobs = get_obs(fiducial,self.observables,self.ells,self.obs_settings,feedback=False).Cls
        print('...done in {:.2f} s'.format(time()-tini))



    def numerical_derivative(self):
        #MMmod: more reliable method to be added here!!! E.g. STEM
        
        derivs   = {param: {f"{ind1}x{ind2}": np.zeros(len(self.ells)) for ind1 in self.cols for ind2 in self.cols}
                    for param in self.param_names }
        
        for i, (param, delta) in enumerate(zip(self.param_names, self.param_deltas)):
            print('...computing derivative for {}...'.format(param))
            fiducial_plus = copy.deepcopy(self.fiducial)
            fiducial_minus = copy.deepcopy(self.fiducial)
            
            
            if param in self.fiducial:
                if self.fiducial[param] == 0:
                    epsilon = delta
                else:
                    epsilon = abs(self.fiducial[param])*delta
                
                fiducial_plus[param]  += epsilon
                fiducial_minus[param] -= epsilon
            
            Cls_plus  = get_obs(fiducial_plus, self.observables, self.ells, self.obs_settings, feedback=False).Cls
            Cls_minus = get_obs(fiducial_minus, self.observables, self.ells, self.obs_settings, feedback=False).Cls
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

        if 'GWWL' in self.observables:
            self.ngwlbin = self.GW_specs['N_gw']/self.Nbins_gwl 
        if 'GWC' in self.observables:
            self.ngwcbin = self.GW_specs['N_gw']/self.Nbins_gwc 
        if 'GC' in self.observables:
            ngalbin_gc = (self.galaxy_specs['gal_per_arcmin']/self.Nbins_gc)*3600*(180/np.pi)**2
        if 'WL' in self.observables:
            ngalbin_wl = (self.galaxy_specs['gal_per_arcmin']/self.Nbins_wl)*3600*(180/np.pi)**2
            eps_error  = self.galaxy_specs['sigma_eps']

        
        calc_obs   = get_obs(self.fiducial, self.observables, self.ells, self.obs_settings,feedback=False)
        noisy_cls  = deepcopy(calc_obs.Cls)
       
        
    
        for ind in (1,self.maxbins):#MMmod: check this when bins are different
            for obs in self.obs:
                if obs == 'G':
                    noisy_cls['G'+str(ind)+'xG'+str(ind)] += 1/ngalbin_gc
                if obs == 'L':
                    noisy_cls['L'+str(ind)+'xL'+str(ind)] += (eps_error**2)/(2*ngalbin_wl)
                if obs == 'WL':
                    noisy_cls['WL'+str(ind)+'xWL'+str(ind)] += [(self.GW_specs['sigma_eps_gw']**2/self.ngwlbin)]
                if obs == 'WC':
                    noisy_cls['WC'+str(ind)+'xWC'+str(ind)] += [(self.GW_specs['sigma_eps_gw']**2/self.ngwcbin)] #Is it the same for GWWL and GWCounts?
        return noisy_cls



    def compute_covmat_second_order(self):
        print('Computing covmat')
        
        noisy_cls=self.get_cls_noisy()
        covmat = []
        
        binrange = range(1, self.maxbins+1 )
        
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
            if 'WL' in self.observables and i<= self.Nbins_wl:
                ngalbin = (self.galaxy_specs['gal_per_arcmin']/self.Nbins_wl)*3600*(180/np.pi)**2
                Nell['L{}xL{}'.format(i,i)] = [(self.galaxy_specs['sigma_eps']**2/(2*ngalbin))]*len(ells)
            if 'GWWL' in self.observables and i<= self.Nbins_gwl:
                self.ngwlbin = self.GW_specs['N_gw']/self.Nbins_gwl 
                Nell['WL{}xWL{}'.format(i,i)] += [(self.GW_specs['sigma_eps_gw']**2/self.ngwlbin)]*len(ells)
            if 'GWC' in self.observables  and i<= self.Nbins_gwc:
                self.ngwcbin = self.GW_specs['N_gw']/self.Nbins_gwc 
                Nell['WC{}xWC{}'.format(i,i)] += [(self.GW_specs['sigma_eps_gw']**2/self.ngwcbin)]*len(ells)

        #MMmod: can we account for different fsky in different probes??
        fsky      = self.galaxy_specs['fsky']
        Delta_ell = self.deltas
        j=0
        obs_list = []
        if 'GC' in self.observables:
            obs_list.append('G')
            j+=1
        if 'WL' in self.observables:
            obs_list.append('L')
            j+=1
        if 'GWWL' in self.observables:
            obs_list.append('WL')
            j+=1
        if 'GWC' in self.observables:
            obs_list.append('WC')
            j+=1

        err_for_cov = np.zeros((j,j,len(ells),self.maxbins,self.maxbins))
        cls_for_cov = np.zeros((j,j,len(ells),self.maxbins,self.maxbins))

        
        

        
        if 'GC' in self.observables and 'WL' in self.observables and 'GWWL' in self.observables and 'GWC' in self.observables:
            for o1,obs1 in enumerate(obs_list):
                if o1==0:
                    Nbins1=self.Nbins_gc
                elif o1==1:
                    Nbins1=self.Nbins_wl
                elif o1==2:
                    Nbins1=self.Nbins_gwl
                elif o1==3:
                    Nbins1=self.Nbins_gwc
                for o2,obs2 in enumerate(obs_list):
                    if o2==0:
                        Nbins2=self.Nbins_gc
                    elif o2==1:
                        Nbins2=self.Nbins_wl
                    elif o2==2:
                        Nbins2=self.Nbins_gwl
                    elif o2==3:
                        Nbins2=self.Nbins_gwc
                    for i in range(Nbins1):
                        for j in range(Nbins2):
                            for ell_ind,ell in enumerate(ells):
                                if obs1 == obs2 and j<i:
                                    Nell[obs1+str(i+1)+'x'+obs2+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                                    self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)] = self.fidobs[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            #print(obs1+str(i+1)+'x'+obs2+str(j+1))

                                err_for_cov[o1,o2,ell_ind,i,j] = Nell[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
                                cls_for_cov[o1,o2,ell_ind,i,j] = self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]

        elif 'GC' in self.observables and 'WL' in self.observables and 'GWWL' in self.observables:
            for o1,obs1 in enumerate(obs_list):
                if o1==0:
                    Nbins1=self.Nbins_gc
                elif o1==1:
                    Nbins1=self.Nbins_wl
                elif o1==2:
                    Nbins1=self.Nbins_gwl
                for o2,obs2 in enumerate(obs_list):
                    if o2==0:
                        Nbins2=self.Nbins_gc
                    elif o2==1:
                        Nbins2=self.Nbins_wl
                    elif o2==2:
                        Nbins2=self.Nbins_gwl
                    for i in range(Nbins1):
                        for j in range(Nbins2):
                            for ell_ind,ell in enumerate(ells):
                                if obs1 == obs2 and j<i:
                                    Nell[obs1+str(i+1)+'x'+obs2+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                                    self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)] = self.fidobs[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            #print(obs1+str(i+1)+'x'+obs2+str(j+1))

                                err_for_cov[o1,o2,ell_ind,i,j] = Nell[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
                                cls_for_cov[o1,o2,ell_ind,i,j] = self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
        
        
        elif 'GC' in self.observables and 'WL' in self.observables and 'GWC' in self.observables:
            for o1,obs1 in enumerate(obs_list):
                if o1==0:
                    Nbins1=self.Nbins_gc
                elif o1==1:
                    Nbins1=self.Nbins_wl
                elif o1==2:
                    Nbins1=self.Nbins_gwc
                for o2,obs2 in enumerate(obs_list):
                    if o2==0:
                        Nbins2=self.Nbins_gc
                    elif o2==1:
                        Nbins2=self.Nbins_wl
                    elif o2==2:
                        Nbins2=self.Nbins_gwc
                    for i in range(Nbins1):
                        for j in range(Nbins2):
                            for ell_ind,ell in enumerate(ells):
                                if obs1 == obs2 and j<i:
                                    Nell[obs1+str(i+1)+'x'+obs2+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                                    self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)] = self.fidobs[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            #print(obs1+str(i+1)+'x'+obs2+str(j+1))

                                err_for_cov[o1,o2,ell_ind,i,j] = Nell[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
                                cls_for_cov[o1,o2,ell_ind,i,j] = self.fidobs[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]


        elif 'GWWL' in self.observables and 'GWC' in self.observables:
            for o1,obs1 in enumerate(obs_list):
                if o1==0:
                    Nbins1=self.Nbins_gwl
                elif o1==1:
                    Nbins1=self.Nbins_gwc
                for o2,obs2 in enumerate(obs_list):
                    if o2==0:
                        Nbins2=self.Nbins_gwl
                    elif o2==1:
                        Nbins2=self.Nbins_gwc
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


            if 'WL' in self.observables: 
                WLcols  = ['L{}xL{}'.format(i,j) for i in range(1,self.Nbins_wl+1) for j in range(i,self.Nbins_wl+1)]
            if 'GC' in self.observables: 
                GCcols  = ['G{}xG{}'.format(i,j) for i in range(1,self.Nbins_gc+1) for j in range(i,self.Nbins_gc+1)]    
            if 'GWWL' in self.observables: 
                GWWLcols  = ['WL{}xWL{}'.format(i,j) for i in range(1,self.Nbins_gwl+1) for j in range(i,self.Nbins_gwl+1)]
            if 'GWC' in self.observables: 
                GWCcols  = ['WC{}xWC{}'.format(i,j) for i in range(1,self.Nbins_gwc+1) for j in range(i,self.Nbins_gwc+1)]
            if 'GC' in  self.observables and 'WL' in self.observables: 
                GGLcols = ['G{}xL{}'.format(i,j) for i in range(1,self.Nbins_gc+1) for j in range(1,self.Nbins_wl+1)]
            if 'WL' in self.observables and 'GWC' in self.observables:
                LGWCcols = ['L{}xWC{}'.format(i,j) for i in range(1,self.Nbins_wl+1) for j in range(1,self.Nbins_gwc+1)]
            if 'GC' in  self.observables and 'GWC' in  self.observables:
                GGWCcols = ['G{}xWC{}'.format(i,j) for i in range(1,self.Nbins_gc+1) for j in range(1,self.Nbins_gwc+1)]
            if 'GC' in  self.observables and 'GWWL' in  self.observables:
                GGWLcols = ['G{}xWL{}'.format(i,j) for i in range(1,self.Nbins_gc+1) for j in range(1,self.Nbins_gwl+1)]
            if 'WL' in self.observables and 'GWWL' in self.observables:
                LGWLcols = ['L{}xWL{}'.format(i,j) for i in range(1,self.Nbins_wl+1) for j in range(1,self.Nbins_gwl+1)]
            if 'GWC' in self.observables and 'GWWL' in self.observables:
                GWCGWLcols = ['WC{}xWL{}'.format(i,j) for i in range(1,self.Nbins_gwc+1) for j in range(1,self.Nbins_gwl+1)]

            all_cols=[]
            if 'WL' in self.observables: 
                all_cols = all_cols+WLcols
            if 'GC' in self.observables:
                if 'WL' in self.observables:
                    all_cols = all_cols+GGLcols+GCcols
                else:
                    all_cols = all_cols+GCcols
            if 'GWC' in self.observables:
                all_cols = all_cols+GWCcols
                if 'GC' in self.observables:
                    all_cols = all_cols+GGWCcols
                if 'WL' in self.observables:
                    all_cols = all_cols+LGWCcols
            if 'GWWL' in self.observables:
                all_cols = all_cols + GWWLcols
                if 'GC' in self.observables:
                    all_cols = all_cols+GGWLcols
                if 'WL' in self.observables:
                    all_cols = all_cols+LGWLcols
                if 'GWC' in self.observables:
                    all_cols = all_cols+GWCGWLcols

            def str_to_ind(in_obs):

                if 'GC' in self.observables and 'WL' in self.observables and 'GWWL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'L':
                        ind = 1
                    elif in_obs == 'WL':
                        ind = 2
                    elif in_obs == 'WC':
                        ind = 3
                elif 'GC' in self.observables and 'WL' in self.observables and 'GWWL' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'L':
                        ind = 1
                    elif in_obs == 'WL':
                        ind = 2
                elif 'GC' in self.observables and 'WL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'L':
                        ind = 1
                    elif in_obs == 'WC':
                        ind = 2
                elif 'GC' in self.observables and 'GWWL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'WL':
                        ind = 1
                    elif in_obs == 'WC':
                        ind = 2
                elif 'WL' in self.observables and 'GWWL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'L':
                        ind = 0
                    elif in_obs == 'WL':
                        ind = 1
                    elif in_obs == 'WC':
                        ind = 2
                elif 'GC' in self.observables and 'WL' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'L':
                        ind = 1
                elif 'GC' in self.observables and 'GWWL' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'WL':
                        ind = 1 
                elif 'WL' in self.observables and 'GWWL' in self.observables:
                    if in_obs == 'L':
                        ind = 0
                    elif in_obs == 'WL':
                        ind = 1 
                elif 'GC' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'G':
                        ind = 0
                    elif in_obs == 'WC':
                        ind = 1 
                elif 'WL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'L':
                        ind = 0
                    elif in_obs == 'WC':
                        ind = 1
                elif 'GWWL' in self.observables and 'GWC' in self.observables:
                    if in_obs == 'WL':
                        ind = 0
                    elif in_obs == 'WC':
                        ind = 1                    
                else:
                    ind = 0

                return ind

            def split_num(s):
                head = s.rstrip('0123456789')
                tail = s[len(head):]
                return head, tail

            for ellind,ell in enumerate(self.fidobs['ells']):
    
                packed_covmat = pd.DataFrame(columns=all_cols,index=all_cols,dtype='float')
    
                for ind1,col in enumerate(all_cols):
                    bin1,bin2 = re.split('x',col)
                    oi1,i1 = split_num(bin1)
                    oj1,j1 = split_num(bin2)
    
                    for ind2,row in enumerate(all_cols):
                        bin1,bin2 = re.split('x',row)
                        oi2,i2 = split_num(bin1)
                        oj2,j2 = split_num(bin2)
            
                        packed_covmat.at[row,col] = covmat[str_to_ind(oi1),str_to_ind(oj1),str_to_ind(oi2),str_to_ind(oj2),
                                                           ellind,int(i1)-1,int(j1)-1,int(i2)-1,int(j2)-1]
            
                #packed_covmat.index = packed_covmat.columns

                invcov = np.linalg.inv(packed_covmat)

                for i1,par1 in enumerate(self.free_params):
                    for i2,par2 in enumerate(self.free_params):

                        der1 = np.array([derivs[par1][col][ellind] for col in all_cols])
                        der2 = np.array([derivs[par2][col][ellind] for col in all_cols])

                        Fisher[ellind,i1,i2] = np.dot(der1,np.dot(invcov,np.transpose(der2)))

            Fisher_final = np.sum(Fisher, axis=0)

        
        return Fisher_final

    
