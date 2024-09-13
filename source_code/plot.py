import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
import re
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from getdist import plots, loadMCSamples, covmat
import getdist
import bios
import sys
from cobaya.samplers.mcmc import plot_progress
from os.path import isfile
import getdist
from getdist import plots, MCSamples, loadMCSamples, covmat
class plotter:
    def __init__(self, fiducial, obs_settings, params, labels, fishers, param_names, fisher_labels):
        self.params = params
        self.fiducial = fiducial
        self.labels = labels
        self.fishers = fishers
        self.obs_settings = obs_settings
        self.param_names = param_names
        self.fisher_labels = fisher_labels

        if len(fishers.keys()) > 1:
            self.samples_list = [self.generate_fake_mcmc_samples(fisher) for fisher in fishers.values()]
        else:
            self.samples = self.generate_fake_mcmc_samples(fishers['fisher'])
        self.plot_getdist_contours()

    def generate_fake_mcmc_samples(self, fisher, num_samples=10000):
        mean = np.array(self.params)
        cov = np.linalg.inv(fisher)
        samples = np.random.multivariate_normal(mean, cov, num_samples)
        return samples

    def print_parameter_bounds(self, mc_samples):
        print('68% Confidence level')
        for param in self.param_names:
            print(mc_samples.getInlineLatex(param, limit=1))

    def analyze_chain(self, name, chain):
        plot_pars = ['omegam', 'H0', 'ns', 'ombh2', 'omch2']
        sample = loadMCSamples(chain['path'], settings={'ignore_rows': chain['burn_in']})
        p = sample.getParams()

        if hasattr(p, 'sigma8') and not hasattr(p, 'S8'):
            p.S8 = p.sigma8 * np.sqrt(p.omegam / 0.3)
            sample.addDerived(p.S8, name='S8', label='S_8')

        sample.cool(chain['temperature'])
        chain['sample'] = sample
        chain['bounds'] = chain['sample'].getTable(paramList=plot_pars, limit=1).tableTex()

        columns = open(chain['path'] + '.1.txt').readline().rstrip().split()
        columns.pop(0)

        points = [pd.read_csv(chain['path'] + '.' + str(i + 1) + '.txt', sep='\s+', skiprows=1, header=None, names=columns)
                  for i in range(chain['Nchains'])]

        chain['trends'] = points

        all_pars = chain['sample'].getParamNames().list()
        primary_pars = chain['sample'].getParamNames().getRunningNames()

        chain['params'] = all_pars
        chain['primary'] = primary_pars

        if chain['get_covmat']:
            covmat = pd.DataFrame(sample.getCov(), columns=all_pars, index=all_pars)
            covmat = covmat.drop([par for par in covmat.columns if par not in primary_pars], axis=1)
            covmat = covmat.drop([par for par in covmat.index if par not in primary_pars], axis=0)

            plt.figure()
            sb.heatmap(covmat, cmap='Reds', cbar_kws={'label': r'$\sigma$'},
                       xticklabels=primary_pars, yticklabels=primary_pars)
            covmat.to_csv(chain['path'] + '_updated_covmat.txt', index=False, header=True, sep='\t')
            chain['covmat'] = covmat

        return chain

    def plot_getdist_contours(self):
        chains = {
            r'MCMC': {
                'path': '/Users/chiaradeleo/Desktop/analysis_and_chains/chains_test_LCDM_3x2pt/3x2pt_LCDM_MH',
                'burn_in': 0.3,
                'Nchains': 4,
                'color': '#ADD8E6',
                'temperature': 10,
                'get_covmat': False
            }
        }

        chains = {name: self.analyze_chain(name, chain) for name, chain in chains.items()}

        if len(self.fishers.keys()) > 1:
            
            selected_markers = {name: self.obs_settings['extra'][name] if name in self.obs_settings['extra']
                                else self.fiducial[name] for name in self.param_names}

            mc_samples_list = [MCSamples(samples=samples, names=self.param_names, labels=self.labels)
                               for samples in self.samples_list]
            g = plots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)
            g.triangle_plot(mc_samples_list, filled=True, contour_colors=['b', 'r'], contour_lws=2,
                            markers=selected_markers, legend_labels=self.fisher_labels)

            plt.savefig("triangle_plot_Fisher_alpha.png", dpi=300, bbox_inches='tight')

        else:
           
            selected_markers = {name: self.obs_settings['extra'][name] if name in self.obs_settings['extra'] else self.fiducial[name] for name in self.param_names}
            
            legend=['Fisher', 'MCMC']
            mc_sample_lists = [MCSamples(samples=self.samples, names=self.param_names, labels=self.labels),
                              *[chain['sample'] for chain in chains.values()]]
            
            g = plots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)
            g.settings.axes_fontsize = 30
            g.settings.axes_labelsize = 30
            g.settings.legend_fontsize = 30
            g.triangle_plot(mc_sample_lists,filled=[True, False],
                            alphas = [0.7, 1], 
                            contour_colors=['cornflowerblue', 'darkorchid'],  # Different colors for different Fisher matrices
                            contour_lws=[2, 3], markers=selected_markers, legend_labels=legend)

            plt.savefig("triangle_plot_Fisher_Planck.png", dpi=300, bbox_inches='tight')
