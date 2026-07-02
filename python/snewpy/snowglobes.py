# -*- coding: utf-8 -*-
"""The ``snewpy.snowglobes`` module contains functions for interacting with SNOwGLoBES.

`SNOwGLoBES <https://github.com/SNOwGLoBES/snowglobes>`_ can estimate detected
event rates from a given input supernova neutrino flux. It supports many
different neutrino detectors, detector materials and interaction channels.
There are three basic steps to using SNOwGLoBES from SNEWPY:
def simulate(SNOwGLoBESdir, flux, detector="all", *, detector_effects=True):
* **Generating input files for SNOwGLoBES:**
    Generate one or more flux or fluence file from a given model using a specified flavor transformation prescription and supernova distance.
    Optional arguments are the name of the output file, an array of time bin edegs, and an array of energy bin edges. If no output name is given,
    a name based on the SN model will be created. The result is a Container object defined in snewpy.flux. 
* **Running SNOwGLoBES:**
    This step takes the flux or fluence Container object and processes it with SNOwGLoBES. The flux / fluence are those generated in the previous step 
    If a SNOwGLoBES detector-type name is given only those detectors are considered: by default all detector types SNOwGLoBES can model are used. 
    The detector_effects argument determines whether the event numbers in the energy bins are 'smeared' or 'unsmeared', and the detector efficiency. 
    The output is a double dictionary of the number of events in a given time bin and energy bin for a given channel in a given detector. 
* **Collating SNOwGLoBES outputs:**
    This step collates together all the interaction channels and time bins evaluated by SNOwGLoBES in a single channel (for each detector and for each time bin).
    The output tables allow to build the detected neutrino energy spectrum and neutrino time distribution, for each reaction channel or the sum of them.
"""

from inspect import isclass
import logging
import os
import re
import tarfile
import importlib

from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u

import snewpy.models
from snewpy.flavor_transformation import *
from snewpy.neutrino import MassHierarchy, MixingParameters
from snewpy.rate_calculator import RateCalculator, center
from snewpy.flux import Container
from snewpy.utils import strip_extensions

logger = logging.getLogger(__name__)

def _get_transformation(flavor_transformation: str):
    """Identify the flavor transformation from a string

    Parameters
    ---------
    flavor_transformation : str
        Name of the flavor transformation

    Returns
    -------
    FlavorTransformation object initialized with default parameters
    """

    IMO_mix_params = MixingParameters(MassHierarchy.INVERTED)
    NMO_mix_params = MixingParameters(MassHierarchy.NORMAL) 
    
    warn("Using a string to specify the flavor transformation is deprecated. Please use a `FlavorTransformation` instance instead.", DeprecationWarning, stacklevel=3)
    if flavor_transformation.startswith(('NeutrinoDecay', 'QuantumDecoherence')):
        print(f"Using default parameters for {flavor_transformation} transformation. Use a `FlavorTransformation` instance to specify custom parameters.")

    # Choose flavor transformation. Use dict to associate the transformation name with its class.
    # The default mixing paramaters are the normal hierarchy values
    flavor_transformation_dict = {'NoTransformation': NoTransformation(), 
                                  'CompleteExchange': CompleteExchange(),                                
                                  'AdiabaticMSW_NMO': AdiabaticMSW(NMO_mix_params), 
                                  'AdiabaticMSW_IMO': AdiabaticMSW(IMO_mix_params), 
                                  'NonAdiabaticMSWH_NMO': NonAdiabaticMSWH(NMO_mix_params), 
                                  'NonAdiabaticMSWH_IMO': NonAdiabaticMSWH(IMO_mix_params), 
                                  'TwoFlavorDecoherence': TwoFlavorDecoherence(NMO_mix_params), 
                                  'TwoFlavorDecoherence_NMO': TwoFlavorDecoherence(NMO_mix_params), 
                                  'TwoFlavorDecoherence_IMO': TwoFlavorDecoherence(IMO_mix_params), 
                                  'ThreeFlavorDecoherence': ThreeFlavorDecoherence(NMO_mix_params),
                                  'NeutrinoDecay_NMO': NeutrinoDecay(NMO_mix_params), 
                                  'NeutrinoDecay_IMO': NeutrinoDecay(IMO_mix_params), 
                                  'QuantumDecoherence_NMO': QuantumDecoherence(NMO_mix_params), 
                                  'QuantumDecoherence_IMO': QuantumDecoherence(IMO_mix_params),
                                  }

    try:
        return flavor_transformation_dict[flavor_transformation]
    except KeyError:
        raise ValueError(f"Flavor transformation '{flavor_transformation}' not found.")

        
def generate(model, flavor_transformation, d, output_filename=None, times=None, energies=None):
    """Generate a flux at a given time or array of fluences for array of time bins, for a given set of energies.
    Flux / fluences will be output into a numpy npz file with either the filename provided or derived from the model name

    Parameters
    ----------
    model : instance of a model class        
    flavor_transformation : str or instance of flavor transformation class  
        If a string, the class is found using the _get_transformation function
    d : astropy Quantity 
        Distance to supernova
    output_filename : str or None
        Stem of output file. If ``None``, will be based on input file name.
        The output file will be a npz file
    times : astropy.Quantity or None
        time to evaluate flux or array of time bin edges over which to compute the fluence
        if None, use the full model time interval for the fluence
    energies : astropy.Quantity or None
        list of energy bin edges at which to compute the flux

    Returns
    -------
    flux container
        defined in snewpy.flux
    """

    # if flavor_transformation is a string, find the appropriate class
    if isinstance(flavor_transformation, str):
        flavor_transformation = _get_transformation(flavor_transformation)

    # set the timings up
    # default if input is None, use full time window of the model
    if times is None:
        times = u.Quantity([model.get_time()[0],model.get_time()[-1]])
    times.sort()                    

    # set up energies
    # default is 0 to 100 MeV in steps of 200 keV
    if energies is None:
        energies = np.linspace(0, 100, 501) << u.MeV
    energies.sort()
    
    # If an array of times are given (or None) inetrgate over time of each interval. 
    # Technically this is a fluence but re-use name
    if len(times) > 1:
        flux = model.get_flux(t=model.get_time(), E=energies, distance=d, flavor_xform=flavor_transformation)
        flux = flux.integrate('time',limits=times)
    else:
        flux = model.get_flux(t=times, E=energies, distance=d, flavor_xform=flavor_transformation)

    #save resulting flux or array of fluences to file
    if output_filename is not None:
        if Path(output_filename).suffix != '.npz':
            flux_filename = output_filename + '.npz'
    else:
        if len(times) > 1:
            flux_filename = f'{model.name}.'+str(flavor_transformation)+f'.{times[0]:.3f}-'+f'{times[-1]:.3f},'+f'{energies[0]:.3f}-'+f'{energies[-1]:.3f},'+f'{d:.3f}'+'.npz'
        else:
            flux_filename = f'{model.name}.'+str(flavor_transformation)+f'.{times:.3f},'+f'{energies[0]:.3f}-'+f'{energies[-1]:.3f},'+f'{d:.3f}'+'.npz'
    flux.save(flux_filename)    
    
    return flux


def simulate(SNOwGLoBESdir, flux, detector="all", *, detector_effects=True):
    """Calculate expected event rates for the given neutrino flux files and the given (set of) SNOwGLoBES detector(s).
    These event rates are given as a function of the neutrino energy and time, for each interaction channel.

    Parameters
    ----------
    SNOwGLoBESdir : str or None
        Path to SNOwGLoBES directory. Set to ``None`` to automatically use the latest supported SNOwGLoBES release.
    flux : str or Flux Container object
        if string, the file of that name will be opened
    detector : str
        Name of detector. If ``"all"``, will use all detectors supported by SNOwGLoBES.
    detector_effects : bool
         Whether to account for detector smearing and efficiency.
         
    Returns
    -------
    dict of dict of flux.Container objects, either dNdT or N
        Dictionary of event rates / numbers: first dict key is detector type, second is channel 
    """

    rc = RateCalculator(base_dir=SNOwGLoBESdir)
    if detector == 'all':
        detector_list = list(rc.detectors)
    elif(isinstance(detector,str)):
        detector_list=[detector]
    else:
        detector_list = detector

    if detector_effects == False:    
        smearing = "unsmeared"
    else:
        smearing = "smeared"
        
    if isinstance(flux,str): #read the flux in the file
        flux_filename = flux
        logging.info(f'Reading fluxes / fluences from {flux_filename}')
        flux = Container.load(flux_filename)
        flux_filename_base = flux_filename[:flux_filename.rfind('.')]        
    else:
        flux_filename = None

    rates = {}            
    for det in detector_list:        
        rates[det] = rc.run(flux, det, detector_effects=detector_effects)
                
    if flux_filename is not None: 
        # save result to file
        if detector == 'all': 
            rates_filename = flux_filename_base+'.all_'+smearing+'.npz'
        else:
            rates_filename = flux_filename_base+'.{detector}_'+smearing+'.npz'
        logging.info(f'Saving detector simulation event rates / numbers to {rates_filenames}')
        np.savez(rates_filename, **{det: np.array(rates[det]) for det in rates})
    
    return rates


def collate(rates):
    """Collates event rates / numbers into distinct channels i.e. add all electron elastic scattering and NC channels

    Parameters
    ----------
    rates : str or dictionary of flux.Container objects, dict key is detector type 
        if str, the file with that name will be opened

    Returns
    -------
    dict of dict of flux.Container objects, either dNdT or N
        Dictionary of event rates / numbers: first dict key is detector type, second is channel 
    """

    def aggregate_channels(rates,patterns):
        for name, pattern in patterns.items():
            #get channels in rates with names that contain the pattern
            matches = [channel for channel in rates.keys() if re.search(pattern,channel)]
            #sum over the matches
            rates_agg = sum(rates[channel] for channel in matches)
            #remove matching channels from rates
            for channel in matches:
                del rates[channel]
            #make a new entry with the aggregate 
            if len(matches) > 0:
                rates[name] = rates_agg
        return rates

    if isinstance(rates,str): #read the flux in the rates_files
        rates_filenames = rates
        logging.info(f'Reading rates from {rates_filenames}')
        rates = np.load(rates_filename)
    else:
        rates_filename = None

    # make collated rate table
    collated_rates = {}
    patterns = {'nc':'nc_','eES':'_e', 
                'coh_helm_Ar':r'coh_helm.*_Ar', 'coh_helm_Ge':r'coh_helm.*_Ge', 'coh_helm_Xe':r'coh_helm.*_Xe',
                'coh_klein-nystrand_Ar':r'coh_klein.*_Ar', 'coh_klein-nystrand_Ge':r'coh_klein.*_Ge', 'coh_klein-nystrand_Xe':r'coh_kelin.*_Xe'                
               }
    for det in rates:
        collated_rates[det] = aggregate_channels(rates[det],patterns)

    if rates_filename is not None:
        # save resulting collated tables to file
        # strip extension of original filename (if present in list of extensions to strip as defined in utils.strip_extensions)
        collated_rates_filename = strip_extensions(rates_filename) + '_collated.npz'
        logging.info(f'Saving collated tables to {rates}')
        np.savez(collated_rates_filename, **{det: np.array(collated_rates[det]) for det in collated_rates})
        
    return collated_rates
