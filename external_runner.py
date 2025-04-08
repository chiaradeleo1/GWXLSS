import sys,os
import yaml
import numpy  as np
import pandas as pd

from source_code.likelihood import LSSlike
from source_code.compute_obs_sources import get_obs

from cobaya.model import get_model
from scipy.stats  import norm
from nautilus     import Prior
from nautilus     import Sampler
    
from bios import read

options = read(sys.argv[1])

num_threads = options['sampler']['nautilus']['num_threads']

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

settings = options['settings']

if options['use_noiseless_cls']:
    data_Cls = pd.read_csv(options['data_path']+'_Cls_noiseless.dat',sep='\s+',header=0)
else:
    data_Cls = pd.read_csv(options['data_path']+'_Cls_noisy.dat',sep='\s+',header=0)

covmat        = np.load(options['data_path']+'_covmat.npy',allow_pickle=True).item()
invcovmat     = {key: np.linalg.inv(val) for key,val in covmat.items()}
distributions = np.load(options['data_path']+'_source_distribution.npy',allow_pickle=True).item()

all_cols = covmat[str(int(data_Cls.iloc[0]['ells']))].columns

fixed_params = {}
print('Preparing the prior...')
prior = Prior()
for par,par_dict in options['params'].items():
    if (type(par_dict) is dict) and ('prior' in par_dict.keys()):

        if len(par_dict['prior'])==2:
           dist_prior = (par_dict['prior']['min'],
                         par_dict['prior']['max'])

        elif par_dict['prior']['dist'] == 'norm':
            dist_prior = norm(loc=par_dict['prior']['loc'],
                              scale=par_dict['prior']['scale'])

        prior.add_parameter(par, dist=dist_prior)
    else:
        fixed_params[par] = par_dict

print('Loaded prior into Nautilus with dimension',prior.dimensionality())
print('Free parameters: ',prior.keys)
print('Fixed parameters: ',list(fixed_params.keys()))

def likelihood_nautilus(param_dict):

    if 'logA' in param_dict:
        param_dict['As'] = np.exp(param_dict.pop('logA'))*1.e-10

    theory = get_obs(param_dict|fixed_params,distributions,data_Cls['ells'],settings).Cls

    chi2 = []
    for ind,ell in enumerate(data_Cls['ells']):
        diffvec = theory.iloc[ind][all_cols].values-data_Cls.iloc[ind][all_cols].values
        chi2.append(np.dot(diffvec,np.dot(invcovmat[str(ell)],diffvec)))

    loglike = -0.5*sum(chi2)

    return loglike

print('Starting to sample with Nautilus...')
nautilus_options = {k:v for k,v in options['sampler']['nautilus'].items() if k != 'num_threads'}
nautilus_options['filepath'] = options['output']+'.hdf5'

sampler = Sampler(prior,likelihood_nautilus,**nautilus_options)

sampler.run(verbose=True)
log_z = sampler.evidence()
points, log_w, log_l = sampler.posterior(equal_weight=True)

params_dict = {par: options['params'][par]['latex'] for par in prior.keys}
nautilus_dict = {'params': {par: options['params'][par] for par in prior.keys}}

with open(options['output']+'.params.yaml', 'w') as outfile:
    yaml.dump(nautilus_dict, outfile, default_flow_style=False)

results = pd.DataFrame(np.c_[points, np.exp(log_w), -log_l],columns=list(params_dict.keys())+['weight','minuslogpost'])
results = results[['weight','minuslogpost']+list(params_dict.keys())]

results.to_csv(options['output']+'_chain.txt',sep='\t',header=False,index=False)
