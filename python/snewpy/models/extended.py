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

        self.__dict__ = base_model.__dict__.copy()
        for method_name in dir(base_model):
            if callable(getattr(base_model, method_name)) and method_name[0] != '_':
                if method_name == 'get_initial_spectra':
                    self._get_initial_spectra = getattr(base_model, method_name)
                else:
                    setattr(self, method_name, getattr(base_model, method_name))
        self.t_final = self.time[-1]
        self.L_final = {flv: self.luminosity[flv][-1] for flv in ThreeFlavor}

        self.k = k
        if A is None:
            A = {}
            tf = self.t_final            
            for flv in ThreeFlavor:
                Lf = self.L_final[flv]
                A[flv] = Lf / (tf.value**k * np.exp(-(tf/tau_c)**alpha))        
        self.A =  A            
        self.tau_c = tau_c
        self.alpha = alpha

    def _get_initial_spectra_dict(self, t, E):
        """Get neutrino spectra/luminosity curves before oscillation
        
        Parameters
        ----------
        t : astropy.Quantity
            Times to add to supernova model.
        E : astropy.Quantity 
            Energies to evaluate the initial spectra.            
        """        
        
        model_spectra = model._get_initial_spectra_dict(self, t, E)
        array = model_spectra
        
        # Select times after the end of the model
        select = t > self.t_final
        L_ext = self.get_extended_luminosity(t)
        
        extended_spectra = {} 
        for flavor in Flavor:
            extended_spectra[flavor] = model_spectra[flavor] * L_ext / L_final[flavor]         
            array[flavor].append(extended_spectra[flavor])
        
        return array

    def get_extended_luminosity(self, t):
        """Get neutrino luminosity from supernova cooling tail luminosity model.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate luminosity.

        Returns
        -------
        astropy.Quantity
            Luminosity calculated from cooling tail model.
        """
        if t.value < 0.5:
            warn("Extended luminosity model not applicable to early times")
        return self.A * t.value**self.k * np.exp(-(t/self.tau_c)**self.alpha)


