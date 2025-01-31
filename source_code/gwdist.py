import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize    import minimize
from scipy.integrate   import quad,trapz
from scipy.special     import erf
import re


class gw_distribution:

    def __init__(self,survey='ET'):

        self.zgw        = np.linspace(0.,5.,1000)
        self.Nz          = self.get_gwdist(survey)
        self.z_bins      = self.get_redshift_bins()
        #print(self.z_bins)
        self.ni_gw          = [interp1d(self.zgw,self.ngw_photoz(self.zgw,i)/trapz(self.ngw_photoz(self.zgw,i),x=self.zgw),kind='cubic',bounds_error=False,fill_value=0.)
                            for i in range(1,len(self.z_bins))]

        self.gwdict = {'dNdz': self.Nz,
                        'bin_lims': self.z_bins,
                        'binned_dist': self.ni_gw}

    def get_gwdist(self,survey):
        dNdz = interp1d(self.zgw,(self.zgw/(1.5))**2*np.exp(-(self.zgw/(1.5))**1.5),bounds_error=False,fill_value=0.)
        match = re.match(r'^ET-(\d+)$', survey) 
        if match:
            self.Nbins = int(match.group(1))  
        else:
            raise ValueError(f"Invalid survey name: {survey}")
        self.zmin  = 0.001
        self.zmax  = 3.

        self.photo = {'fout': 0.1,
                        'co': 1,
                        'cb': 1,
                        'sigma_o': 0.05,
                        'sigma_b': 0.05,
                        'zo': 0.1,
                        'zb': 0.}

        return dNdz


    def ngw_photoz(self, z, i):

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

    

        return np.array(zbins)

        ##MMmod: here we could add a check that the binned output makes sense
        ##       in general, this function to find the bin can be improved!


