import astropy.units as u
import numpy as np
from snewpy.models import pisn
from snewpy.neutrino import MassHierarchy, MixingParameters
from snewpy.flavor_transformation import AdiabaticMSW
from snewpy.rate_calculator import RateCalculator
import pytest

pytestmark=pytest.mark.snowglobes

rc = RateCalculator()

distance = 1000*u.pc
 #SNOwGLoBES detector for water Cerenkov
E = np.linspace(0,20,100)*u.MeV

@pytest.mark.parametrize('model_class,model_params',[
    (pisn.Wright_2017, {'progenitor_mass': 250*u.Msun, 'eos': 'Helm'}),
])
@pytest.mark.parametrize('transformation',[AdiabaticMSW(MixingParameters(mh)) for mh in MassHierarchy])

def test_pisn_rate(model_class, model_params, transformation):
    model = model_class(**model_params)
    fluence = model.get_fluence(E, distance=distance, flavor_xform=transformation)
    dNdE = rc.run(fluence, detector='wc100kt30prct', detector_effects=False)['ibd']
    #the factor of 0.5 is because the mass of SuperK is 50kt, not 100kt
    Nevents = 0.5 * dNdE.integrate_or_sum('energy').array.squeeze()
    assert 1<Nevents<10
