import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
import getdist
from getdist import plots, MCSamples

class plotter:

    def __init__(self, fiducial, obs_settings, params, labels, fishers, param_names, fisher_labels):
        self.params = params
        self.fiducial = fiducial
        self.labels = labels
        self.fishers = fishers
        self.obs_settings=obs_settings
        
        self.param_names = param_names
        self.fisher_labels = fisher_labels
        print(len(fishers.keys()))
        if len(fishers.keys())>1:
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

    def plot_getdist_contours(self):
        if len(self.fishers.keys())>1:
            selected_markers = {name: self.obs_settings['extra'][name] if name in self.obs_settings['extra'] else self.fiducial[name] for name in self.param_names}
            mc_samples_list = [MCSamples(samples=samples, names=self.param_names, labels=self.labels) for samples in self.samples_list]
            g = plots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)
            g.settings.figure_legend_frame = False
            g.settings.axes_fontsize = 30
            g.settings.axes_labelsize = 30
            g.settings.legend_fontsize = 30
            g.settings.axis_marker_color = 'black'
            g.settings.axis_marker_ls = '--'
            g.settings.axis_marker_lw = 3
            g.settings.axis_tick_x_rotation = 45
    
            # Overlay the contours
            g.triangle_plot(mc_samples_list,
                            filled=[True, False],
                            alphas = [0.7, 1], 
                            contour_colors=['cornflowerblue', 'darkorchid'],  # Different colors for different Fisher matrices
                            contour_lws=[2, 3],
                            markers=selected_markers,
                            legend_labels=self.fisher_labels)
    
            g.fig.align_ylabels()
            g.fig.align_xlabels()
            #plt.legend(self.fisher_labels)
            plt.savefig("triangle_plot_Fisher_alpha.png", dpi=300, bbox_inches='tight')

        else:
            
            selected_markers = {name: self.obs_settings['extra'][name] if name in self.obs_settings['extra'] else self.fiducial[name] for name in self.param_names}
            self.mc_samples = MCSamples(samples=self.samples, names=self.param_names, labels=self.labels)
            g = plots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)
            g.settings.figure_legend_frame = False
            g.settings.axes_fontsize = 20
            g.settings.axes_labelsize = 20
            g.settings.legend_fontsize = 20
            g.settings.axis_marker_color = 'black'
            g.settings.axis_marker_ls = '--'
            g.settings.axis_marker_lw = 3
            g.settings.axis_tick_x_rotation = 45
            g.triangle_plot(self.mc_samples,
                            filled=True,
                            legend_loc='upper right',
                            contour_lws=2,
                            markers=selected_markers)
            g.fig.align_ylabels()
            g.fig.align_xlabels()
            self.print_parameter_bounds(self.mc_samples)
            plt.savefig("triangle_plot_Fisher_Planck.png", dpi=300, bbox_inches='tight')
