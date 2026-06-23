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
        Name of output file. If ``None``, will be based on input file name.
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
    times = None
    if tstart is not None and tend is not None:
        try:
            #in case we have arrays: join them together
            times = np.append(tstart, tend)
            #and get rid of the duplicates with 1e-10 tolerance
            times = np.unique(times.round(decimals=10))
        except:
            #in case we have single values
            times = u.Quantity([tstart,tend])
        times.sort()
    else:
        times = model.get_time()

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

    ###energies_t = (np.linspace(0, 100, 201) + 0.25 ) << u.MeV 

    flux = model.get_flux(t=times, E=energies, distance=d, flavor_xform=flavor_transformation)
    fluence = flux.integrate('time', limits = times).integrate('energy', limits = energies)        

    #save resulting fluence to file
    if output_filename is not None:
        tfname = output_filename + '.npz'
    else: # strip extension (if present in list of extensions to strip as defined in utils.strip_extensions)
        model_file_root = strip_extensions(model.filename) 
        tfname = f'{model_file_root},'+str(flavor_transformation)+f',{times[0]:.3f}-'+f'{times[-1]:.3f},'+f'{energies[0]:.3f}-'+f'{energies[-1]:.3f},'+f'{d:.3f}'+'.npz'
    ##fluence.save(tfname)
    flux.save(tfname)    
    
    return tfname



def simulate(SNOwGLoBESdir, tarball_path, detector_input="all", *, detector_effects=True):
    """Calculate expected event rates for the given neutrino flux files and the given (set of) SNOwGLoBES detector(s).
    These event rates are given as a function of the neutrino energy and time, for each interaction channel.

    Parameters
    ----------
    SNOwGLoBESdir : str or None
        Path to SNOwGLoBES directory. Set to ``None`` to automatically use the latest supported SNOwGLoBES release.
    tarball_path : str
        Path of compressed .tar file produced e.g. by ``generate_time_series()`` or ``generate_fluence()``.
    detector_input : str
        Name of detector. If ``"all"``, will use all detectors supported by SNOwGLoBES.
    detector_effects : bool
         Whether to account for detector smearing and efficiency.
    """
    rc = RateCalculator(base_dir=SNOwGLoBESdir)
    if detector_input == 'all':
        detector_input = list(rc.detectors)
    if(isinstance(detector_input,str)):
        detector_input=[detector_input]
    rates_dict = {}

    if detector_effects == False:    
        smearing = "unsmeared"
    else:
        smearing = "smeared"
        
    #read the fluence
    flux = Container.load(tarball_path)
    
    for det in detector_input:
        rates=rc.run(flux, det, detector_effects=detector_effects)
        #collect everything to pandas DataFrame, to make the output similar to previous
        rates_dict[det]={'weighted':{smearing : rates}}
        
    # reorder results to produce the same format as before:
    #    {detector: {time_bin:{'weighted':{smeared/unsmeared: [rate vs energy bins]}}}}
    result = {}
    fname_base = tarball_path[:tarball_path.rfind('.')]
    for det in rates_dict:
        #get the time bins
        rates = rates_dict[det]['weighted'][smearing]

        #get the first rate from the dict to access the energy and time binning
        some_rate = list(rates.values())[0]
        tbins = center(some_rate.time)
        ebins = center(some_rate.energy)
        result[det] = {}
        for n_bin, t_bin in enumerate(tbins):
            data = {**{(chan,smearing,'weighted'): rate.array[0,n_bin,:]
                      for chan,rate in rates.items()} }
            
            df = pd.DataFrame(data, index = ebins)
            df.index.rename('E', inplace=True)
            df.columns.rename(['channel','is_'+smearing,'is_weighted'], inplace=True)            
            df = df.reorder_levels([2,1,0], axis='columns')
            if len(tbins) > 1:
                result[det][f'{fname_base}_{n_bin:01d}'] = df
            else:
                result[det][f'{fname_base}'] = df
        
    # save result to file for re-use in collate()
    cache_file = f'{fname_base}.'+smearing+'.npy'
    logging.info(f'Saving simulation results to {cache_file}')
    np.save(cache_file, result)
    
    return result


re_chan_label = re.compile(r'nu(e|mu|tau)(bar|)_([A-Z][a-z]*)(\d*)_?(.*)')
def get_channel_label(c):
    mapp = {'nc':'NeutralCurrent',
            'ibd':'Inverse Beta Decay',
            'e':r'${\nu}_x+e^-$'}
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

def collate(tarball_path):
    """Collates SNOwGLoBES output files and generates plots or returns a data table.

    Parameters
    ----------
    tarball_path : str
        Path of compressed .tar file produced e.g. by generate.

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

    #read the results from storage
    cache_file = tarball_path[:tarball_path.rfind('.')] + '.npy'
    cache_file_stem, smearing = cache_file.rsplit('.',1)
    
    logging.info(f'Reading tables from {cache_file}')
    tables = np.load(cache_file, allow_pickle=True).tolist()
    #This output is similar to what produced by:
    #tables = simulate(SNOwGLoBESdir, tarball_path,detector_input)

    #dict for old-style results, for backward compatibiity
    results = {}
    #smearing_options = ['smeared','unsmeared'] if smearing else ['unsmeared']
    #save collated files:
    with TemporaryDirectory(prefix='snowglobes') as tempdir:
        tempdir = Path(tempdir)
        for det in tables:
            results[det] = {}
            for flux,t in tables[det].items():
                t = aggregate_channels(t,nc='nc_',e='_e')

                table = t['weighted'][smearing]
                filename_base = f'{flux}_{det}_events_{smearing}_{'weighted'}'
                filename = tempdir/f'Collated_{filename_base}.dat'
            #save results to text files
            with open(filename,'w') as f:
                f.write(table.to_string(float_format='%23.15g'))
                #format the results for the output
                header = 'Energy '+' '.join(list(table.columns))
                data = table.to_numpy().T
                index = table.index.to_numpy()
                data = np.concatenate([[index],data])
                results[filename.name] = {'header':header,'data':data}
 
        #Make a tarfile with the condensed data files and plots
        output_name = Path(tarball_path).stem
        output_name = output_name[:output_name.rfind('.tar')]+'_SNOprocessed'
        output_path = Path(tarball_path).parent/(output_name+'.tar.gz')
        with tarfile.open(output_path, "w:gz") as tar:
            for file in tempdir.iterdir():
                tar.add(file,arcname=output_name+'/'+file.name)
        logging.info(f'Created archive: {output_path}')
    return results 
