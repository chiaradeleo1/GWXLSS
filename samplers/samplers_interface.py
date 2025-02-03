import sys,os
import yaml
import numpy  as np
import pandas as pd

def nautilus_interface(info):

    #done by GCH. Ask how to credit
    from cobaya.model import get_model
    from scipy.stats  import norm
    from nautilus     import Prior
    from nautilus     import Sampler

    num_threads = info['sampler']['nautilus']['num_threads']

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    print('')
    print('RUNNING WITH NAUTILUS SAMPLER')
    print('')

    print('Loading model wrapper of Cobaya')
    model = get_model(info)
    print('model loaded')
    point = dict(zip(model.parameterization.sampled_params(),
                     model.prior.sample(ignore_external=True)[0]))
    logposterior = model.logposterior(point)

    print('Preparing the prior...')
    prior = Prior()

    for par in info['params'].keys():
        if (type(info['params'][par]) is dict) and ('prior' in info['params'][par].keys()):

            if len(info['params'][par]['prior'])==2:
                dist_prior = (info['params'][par]['prior']['min'],
                              info['params'][par]['prior']['max'])

            elif info['params'][par]['prior']['dist'] == 'norm':
                dist_prior = norm(loc=info['params'][par]['prior']['loc'],
                                  scale = info['params'][par]['prior']['scale'])

            prior.add_parameter(par, dist=dist_prior)

    print('Loaded prior into Nautilus with dimension',prior.dimensionality())
    print('Prior keys: ',prior.keys)

    derived_pars = [k for k in info['params'].keys() if type(info['params'][k]) == dict and 'prior' not in info['params'][k]]+['chi2']
    blob_vec     = [(par, float) for par in derived_pars]

    ## Likelihood
    def likelihood_nautilus(param_dict):

        logpost        = model.logposterior(param_dict)
        derived_params = logpost.derived
        chi2           = -2*logpost.loglike


        like_tuple    = [model.logposterior(param_dict).loglike]+[par for par in derived_params]+[chi2]
        full_tuple    = tuple(like_tuple)

        return full_tuple 


    print('Starting to sample with Nautilus...')
    nautilus_options = {k:v for k,v in info['sampler']['nautilus'].items() if k != 'num_threads'}
    if 'output' in info:
        nautilus_options['filepath'] = info['output']+'.hdf5'

    sampler = Sampler(prior,likelihood_nautilus,**nautilus_options,blobs_dtype=blob_vec)

    sampler.run(verbose=True)
    log_z = sampler.evidence()
    points, log_w, log_l, derived = sampler.posterior(equal_weight=True,return_blobs=True)
    derived_array = np.array([np.array(list(der)) for der in derived])

    params_dict = {par: info['params'][par]['latex'] for par in info['params'] if type(info['params'][par]) == dict} | {'chi2': '\chi^2'}
    nautilus_dict = {'params': {par: info['params'][par] for par in info['params'] if type(info['params'][par]) == dict} | {'chi2': {'latex': '\chi^2'}}} | {'theory': info['theory']}

    if 'output' in info:
        with open(info['output']+'.params.yaml', 'w') as outfile:
            yaml.dump(nautilus_dict, outfile, default_flow_style=False)

    results = pd.DataFrame(np.c_[points, derived_array, np.exp(log_w), -log_l],columns=list(params_dict.keys())+['weight','minuslogpost'])
    results = results[['weight','minuslogpost']+list(params_dict.keys())]

    if 'output' in info:
        results.to_csv(info['output']+'_chain.txt',sep='\t',header=False,index=False)


    print('NAUTILUS SAMPLING FINISHED')

    return results

def run_fisher(info):

    possible_observables = ['GC','WL','GWC', 'GWWL' ]
    
    lmin = np.log10(info['analysis_settings']['lmin'])
    lmax = np.log10(info['analysis_settings']['lmax'])
    N    = info['analysis_settings']['Nbin_ell']

    obs_settings = info['obs_settings']

    if obs_settings['extra'] == None:
        obs_settings['extra'] = {}

    ell_lims = np.logspace(lmin,lmax,N) #creation of array-> N bin log spaced
    ells     = np.array([int(ell) for ell in 0.5*(ell_lims[:-1]+ell_lims[1:])])
    deltas   = (ell_lims[1:]-ell_lims[:-1]) #evaluation of the amplitude of each bin

    distributions = np.load(info['analysis_settings']['dist_path'],allow_pickle=True).item()
    for obs in distributions.keys():
            if obs not in possible_observables:
                sys.exit( "Unknown observable in source distribution file: {}. Possible observables are: "
                "photometric Galaxy clustering (GC), galaxy Weak Lensing (WL), "
                "Gravitational Waves Weak Lensing (GWWL), and Gravitational Waves Counts (GWC)".format(obs))

    galaxy_specs  = info['analysis_settings']['galaxy_specs']

    #MMmod: to be changed to work only with GW only?#####
    if 'GWWL' in distributions or 'GWC' in distributions:
        GW_specs = info['analysis_settings']['GW_specs']
    else:
        GW_specs = {}
    #####################################################
    
    free_params = info['sampler']['Fisher']['freepars']

    fiducial = {par: pardict['fiducial'] for par,pardict in free_params.items()} | info['sampler']['Fisher']['fixedpars']
    dertype  = info['sampler']['Fisher']['derivative'] 

    from source_code.fisher import get_Fisher
    fishmodule = get_Fisher(fiducial,free_params,ells,deltas,distributions,obs_settings,galaxy_specs,GW_specs,dertype)
    fishmat    = fishmodule.fisher_matrix()

    fisher = pd.DataFrame(fishmat,columns=free_params.keys(),index=free_params.keys())
    params_info = {par: val for par,val in free_params.items()}


    return fisher,params_info
