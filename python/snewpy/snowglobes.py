# -*- coding: utf-8 -*-
"""The ``snewpy.snowglobes`` module contains functions for interacting with SNOwGLoBES.

`SNOwGLoBES <https://github.com/SNOwGLoBES/snowglobes>`_ can estimate detected
event rates from a given input supernova neutrino flux. It supports many
different neutrino detectors, detector materials and interaction channels.
There are three basic steps to using SNOwGLoBES from SNEWPY:

* **Generating input files for SNOwGLoBES:**
    There are two ways to do this, either generate a time series or a fluence file. This is done taking as input the supernova simulation model.
    The first will evaluate the neutrino flux at each time step, the latter will compute the integrated neutrino flux (fluence) in the time bin.
    The result is a compressed .tar file containing all individual input files.
* **Running SNOwGLoBES:**
    This step convolves the fluence generated in the previous step with the cross-sections for the interaction channels happening in various detectors supported by SNOwGLoBES.
    It takes into account the effective mass of the detector as well as a smearing matrix describing the energy-dependent detection efficiency.
    The output gives the number of events detected as a function of energy for each interaction channel, integrated in a given time window (or time bin), or in a snapshot in time.
* **Collating SNOwGLoBES outputs:**
    This step puts together all the interaction channels and time bins evaluated by SNOwGLoBES in a single file (for each detector and for each time bin).
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


def generate(model, flavor_transformation, d, output_filename=None, tstart=None, tend=None, Emin=None, Emax=None):
    """Generate time series files in SNOwGLoBES format.

    This version will subsample the times in a supernova model, produce energy
    tables expected by SNOwGLoBES, and compress the output into a tarfile.

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
    tstart : astropy.Quantity or None
        Start of time interval to integrate over, or list of start times of the time series bins.
    tend : astropy.Quantity or None
        End of time interval to integrate over, or list of end times of the time series bins.        
    Emin : astropy.Quantity or None
        Minimum energy for the spectrum, or list of minimum energy for energy bins
    Emax : astropy.Quantity or None    
        Maximum energy for the spectrum, or list of maximum energy for energy bins    
        

    Returns
    -------
    str
        Path of NumPy archive file with neutrino fluence data.
    """

    # if flavor_transformation is a string, find the appropriate class
    if isinstance(flavor_transformation, str):
        flavor_transformation = _get_transformation(flavor_transformation)

    # set the timings up
    # default if inputs are None: full time window of the model
    times = np.array([])
    if tstart is not None:
        try: 
            times.append(tstart)
        except: #in case we have single values
            times.append(u.Quantity([tstart]))
    else:
        model_times = model.get_time()
        times.append(u.Quantity([model_times[0]]))

    if tend is not None:
        try: 
            times.append(tend)
        except: #in case we have single values
            times.append(u.Quantity([tend]))
    else:
        model_times = model.get_time()
        times.append(u.Quantity([model_times[-1]]))

    times.sort()
    #get rid of the duplicates with 1e-10 tolerance
    times = np.unique(np.round(times,decimals=10))

    # set up energies
    # default is 0 to 100 MeV in steps of 200 keV
    energies = None    
    if Emin is not None and Emax is not None:
        try:
            #in case we have arrays: join them together
            energies = np.append(Emin, Emax)
            #and get rid of the duplicates with 1e-10 tolerance
            energies = np.unique(energies.round(decimals=10))
        except:
            #in case we have single values
            energies = u.Quantity([Emin,Emax])
    else:        
        energies = np.linspace(0, 100, 501) << u.MeV

    flux = model.get_flux(t=times, E=energies, distance=d, flavor_xform=flavor_transformation)

    #save resulting flux to file
    if output_filename is not None:
        flux_filename = output_filename + '.npz'
    else: # strip extension (if present in list of extensions to strip as defined in utils.strip_extensions)
        flux_filename_root = strip_extensions(model.filename) 
        flux_filename = f'{flux_filename_root},'+str(flavor_transformation)+f',{times[0]:.3f}-'+f'{times[-1]:.3f},'+f'{energies[0]:.3f}-'+f'{energies[-1]:.3f},'+f'{d:.3f}'+'.npz'
    flux.save(flux_filename)    
    
    return flux, flux_filename


def simulate(SNOwGLoBESdir, flux_filename, detector="all", *, detector_effects=True):
    """Calculate expected event rates for the given neutrino flux files and the given (set of) SNOwGLoBES detector(s).
    These event rates are given as a function of the neutrino energy and time, for each interaction channel.

    Parameters
    ----------
    SNOwGLoBESdir : str or None
        Path to SNOwGLoBES directory. Set to ``None`` to automatically use the latest supported SNOwGLoBES release.
    flux_filename : str
        Path of npz file produced by generate.
    detector : str
        Name of detector. If ``"all"``, will use all detectors supported by SNOwGLoBES.
    detector_effects : bool
         Whether to account for detector smearing and efficiency.
    """
    rc = RateCalculator(base_dir=SNOwGLoBESdir)
    if detector == 'all':
        detector_list = list(rc.detectors)
    if(isinstance(detector,str)):
        detector_list=[detector]

    if detector_effects == False:    
        smearing = "unsmeared"
    else:
        smearing = "smeared"
        
    #read the flux in the flux_file
    flux = Container.load(flux_filename)

    rates = {}
    for det in detector_list:        
        rates[det] = rc.run(flux, det, detector_effects=detector_effects)
                
    # save result to file for re-use in collate()
    fname_base = flux_filename[:flux_filename.rfind('.')]               
    if detector == 'all':
        rates_filename = f'{fname_base}.all'+smearing+'.npy'        
    else:
        rates_filename = f'{fname_base}.{detector}_'+smearing+'.npy'
        
    logging.info(f'Saving detector simulation event rates to {rates_filename}')
    np.save(rates_filename, rates)
    
    return rates, rates_filename


re_chan_label = re.compile(r'nu(e|mu|tau)(bar|)_([A-Z][a-z]*)(\d*)_?(.*)')
def get_channel_label(c):
    mapp = {'nc':'NeutralCurrent',
            'ibd':'Inverse Beta Decay',
            'eES':r'${\nu}_x+e^-$'}
    def gen_label(m):
        flv,bar,Nuc,num,res = m.groups()
        if flv!='e':
            flv='\\'+flv
        if bar:
            bar='\\'+bar
        s = f'${bar}{{\\nu}}_{flv}$ '+f'${{}}^{{{num}}}{Nuc}$ '+res
        return s

    if c in mapp:
        return mapp[c]
    else: 
        return re_chan_label.sub(gen_label, c) 

def collate(rates_filename):
    """Collates SNOwGLoBES output files and generates plots or returns a data table.

    Parameters
    ----------
    rates_filename : str
        File with cached rates from simulate 

    Returns
    -------
    dict
        Dictionary of data tables: One table per time bin; each table contains in the first column the energy bins, in the remaining columns the number of events for each interaction channel in the detector.
    """

    def aggregate_channels(table, **patterns):
        #rearrange the table to have only channel column
        levels = list(table.columns.names)
        levels.remove('channel')
        if pd.__version__ < '2.1':
            t = table.stack(levels)
        else:
            # Avoid FutureWarning, see https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#new-implementation-of-dataframe-stack
            t = table.stack(levels, future_stack=True)
        for name,pattern in patterns.items():
            #get channels which contain `like`
            t_sel = t.filter(like=pattern)
            #sum over them and save to a separate column
            t_agg = t_sel.sum(axis='columns')
            #drop processed channels
            t.drop(t_sel.columns, axis='columns',inplace=True)
            t[name]=t_agg #fill the column
        #return table with the original levels order
        t = t.unstack(levels)
        t = t.reorder_levels(table.columns.names, axis=1)
        return t
        
    #read the results from rates_file produced by simulate(SNOwGLoBESdir,tarball_path,detector_input)    
    logging.info(f'Reading tables from {rates_filename}')
    tables = np.load(rates_filename, allow_pickle=True).tolist()

    collated_tables = {}
    #make collated tables and save:
    for det in tables:
        collated_tables[det] = {}
        for flux,table in tables[det].items():
            table = aggregate_channels(table,nc='nc_',eES='_e')
            collated_tables[det][flux] = {table}

    # save resulting collated tables to file
    # strip extension (if present in list of extensions to strip as defined in utils.strip_extensions)
    collated_rates_filename = strip_extensions(rates_filename) + '_collated.npz'
    logging.info(f'Saving collated tables to {rates}')
    np.save(collated_rates_filename, collated_tables)
        
    return collated_tables, collated_rates_filename 
