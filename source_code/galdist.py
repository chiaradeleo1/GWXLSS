import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize    import minimize
from scipy.integrate   import quad,trapz
from scipy.special     import erf

class galaxy_distribution:

    def __init__(self,survey='Euclid'):

        self.zgal        = np.linspace(0.,5.,1000)
        self.Nz          = self.get_galdist(survey)
        self.z_bins      = self.get_redshift_bins()
        #print(self.z_bins)
        self.ni          = [interp1d(self.zgal,self.ngal_photoz(self.zgal,i)/trapz(self.ngal_photoz(self.zgal,i),x=self.zgal),kind='cubic',bounds_error=False,fill_value=0.)
                            for i in range(1,len(self.z_bins))]

        self.galdict = {'dNdz': self.Nz,
                        'bin_lims': self.z_bins,
                        'binned_dist': self.ni}

    def get_galdist(self,survey):

        if survey == 'Euclid-10':
            dNdz = interp1d(self.zgal,(self.zgal/(0.9/np.sqrt(2)))**2*np.exp(-(self.zgal/(0.9/np.sqrt(2)))**1.5),bounds_error=False,fill_value=0.)
            self.Nbins = 10
            self.zmin  = 0.001
            self.zmax  = 3.

            self.photo = {'fout': 0.1,
                          'co': 1,
                          'cb': 1,
                          'sigma_o': 0.05,
                          'sigma_b': 0.05,
                          'zo': 0.1,
                          'zb': 0.}

        elif survey == 'Euclid-13':
            dNdz = interp1d(self.zgal,(self.zgal/(0.9/np.sqrt(2)))**2*np.exp(-(self.zgal/(0.9/np.sqrt(2)))**1.5),bounds_error=False,fill_value=0.)
            self.Nbins = 13
            self.zmin  = 0.001
            self.zmax  = 3.

            self.photo = {'fout': 0.1,
                          'co': 1,
                          'cb': 1,
                          'sigma_o': 0.05,
                          'sigma_b': 0.05,
                          'zo': 0.1,
                          'zb': 0.}

        else:
            print('not there yet')

        return dNdz

    def ngal_photoz(self, z, i):

        if i == 0 or i >= self.Nbins+1:
            return None

        term1 =self.photo['cb']*self.photo['fout']*erf((0.707107*(z-self.photo['zo']-self.photo['co']*self.z_bins[i - 1]))/(self.photo['sigma_o']*(1+z)))
        term2 =-self.photo['cb']*self.photo['fout']*erf((0.707107*(z-self.photo['zo']-self.photo['co']*self.z_bins[i]))/(self.photo['sigma_o']*(1+z)))
        term3 =self.photo['co']*(1-self.photo['fout'])*erf((0.707107*(z-self.photo['zb']-self.photo['cb']*self.z_bins[i - 1]))/(self.photo['sigma_b']*(1+z)))
        term4 =-self.photo['co']*(1-self.photo['fout'])*erf((0.707107*(z-self.photo['zb']-self.photo['cb']*self.z_bins[i]))/(self.photo['sigma_b']*(1+z)))

        return self.Nz(z)*(term1+term2+term3+term4)/(2*self.photo['co']*self.photo['cb'])


    def get_redshift_bins(self):

        zint = np.linspace(self.zmin,self.zmax,10000)
        dNdz = interp1d(zint,self.Nz(zint)/trapz([self.Nz(z) for z in zint],x=zint))
        pbin = 1./self.Nbins

        inf = self.zmin
        sup = self.zmax

        zbins = [inf]

        for _ in range(self.Nbins):

            def bin_prob(x):
                area = quad(dNdz,inf,x)[0]
                return abs(area-pbin)

            res = minimize(bin_prob, 0.5*(inf+sup), method='L-BFGS-B',bounds=[(inf,self.zmax)])
            inf = res.x[0]

            zbins.append(res.x[0])

    #def get_redshift_bins(self):

     #   centers = [0.5, 1.5, 3.0]
      #  sigma = 0.2
       # zbins = []

        #for center in centers:
         #   bin_min = center - sigma
          #  bin_max = center + sigma
           # zbins.append(bin_min)
            #zbins.append(bin_max)

        return np.array(zbins)

        ##MMmod: here we could add a check that the binned output makes sense
        ##       in general, this function to find the bin can be improved!


