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

        """self.__dict__ = base_model.__dict__.copy()
        for method_name in dir(base_model):
            if callable(getattr(base_model, method_name)) and method_name[0] != '_':
                if method_name == 'get_initial_spectra':
                    self._get_initial_spectra = getattr(base_model, method_name)
                else:
                    setattr(self, method_name, getattr(base_model, method_name))"""
        self.model = base_model
        super().__init__(model.time,model.metadata)        

        self.k = k
        if A is None:
            A = 1 / ( self.time[-1].value**k * np.exp(-(self.time[-1]/tau_c)**alpha) ) 
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
        
        model_spectra = self.model._get_initial_spectra_dict(self, t, E)
                
        # Select times after the end of the model
        t_ext = t > self.t_final
        f_ext = self.get_extended_time_dependence(t)
        
        array = {} 
        for flavor in flavors:
            array[flavor] = model_spectra[flavor]
            extended_spectra[flavor] = np.outer( f_ext, model_spectra[flavor][-1,:])
            array[flavor].append(extended_spectra[flavor])
        
        return array

    def get_extended_time_dependence(self, t):
        """Get neutrino luminosity from supernova cooling tail luminosity model.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate luminosity.

        Returns
        -------
        astropy.Quantity
            extended time dependence calculated from cooling tail model.
        """
        if t.value < 0.5:
            warn("Extended luminosity model not applicable to early times")
        return self.A * t.value**self.k * np.exp(-(t/self.tau_c)**self.alpha)


