from warnings import warn

import numpy as np
from astropy import units as u

from snewpy.flavor import ThreeFlavor
from snewpy.models.base import SupernovaModel

class ExtendedModel(SupernovaModel):
    """Class defining a supernova model with a cooling tail extension."""

    def __init__(self, base_model, k=-1., A=None, tau_c=36. * u.s, alpha=2.66):
        """Initialize extended supernova model class."""
        if not isinstance(base_model, SupernovaModel):
            raise TypeError("ExtendedModel.__init__ requires a SupernovaModel object")

        self.base_model = base_model
        super().__init__(base_model.time,base_model.metadata)        

        self.k = k
        if A is None:
            A = 1 / ( self.time[-1]**k * np.exp(-(self.time[-1]/tau_c)**alpha) ) 
        self.A =  A            
        self.tau_c = tau_c
        self.alpha = alpha

    def _get_initial_spectra_dict(self, t, E, flavors=ThreeFlavor):
        """Get neutrino spectra/luminosity curves before oscillation
        
        Parameters
        ----------
        t : astropy.Quantity
            Times to add to supernova model.
        E : astropy.Quantity 
            Energies to evaluate the initial spectra.            
        """        
        #convert input arguments to 1D arrays
        t = u.Quantity(t, ndmin=1)
        E = u.Quantity(E, ndmin=1)        
        
        base_model_spectra = self.base_model._get_initial_spectra_dict(t, E)
                
        # Select times after the end of the model
        t_ext = t > self.time[-1]
        f_ext = self.get_extended_time_dependence(t_ext)
        
        array = {} 
        for flavor in flavors:
            array[flavor] = base_model_spectra[flavor]
            extended_model_spectra[flavor] = np.outer( f_ext, base_model_spectra[flavor][-1,:])
            array[flavor].append(extended_model_spectra[flavor])
        
        return array

    def get_extended_time_dependence(self, times):
        """Get neutrino luminosity from supernova cooling tail luminosity model.

        Parameters
        ----------
        times : astropy.Quantity
            Times to evaluate luminosity.

        Returns
        -------
        astropy.Quantity
            extended time dependence calculated from cooling tail model.
        """
        if times[0] < 0.5*u.s:
            warn("Extended luminosity model not applicable to early times")
        f = np.empty(len(times))
        for i in range(len(times)):
            f[i] = self.A * times[i]**self.k * np.exp(-(times[i]/self.tau_c)**self.alpha)
        return f


