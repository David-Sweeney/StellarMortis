import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ebf
from galpy import potential
from galpy.potential.mwpotentials import MWPotential2014
from galpy.orbit import Orbit
from galpy.util.conversion import get_physical
from astropy import units as u

from .kick import NatalKick
from .mass import Mass

def calculate_lifetimes(df):
    """
    Function to calculate the lifetime of each star.

    Adapted from code provided by Sanjib Sharma.
    """
    with open('../data/stellar_lifetime.pkl', 'rb') as handle:
        stellar_lifetimes = pickle.load(handle)
    stellar_properties = np.array([np.clip(df['feh'], a_min=-2, a_max=0.49), df['smass']]).T
    df['lifetime'] = 10**stellar_lifetimes(stellar_properties) / 10**9
    return df

def load_data(filename):
    '''Load Galaxia data into DataFrame'''
    data = ebf.read(filename, '/')
    centre = np.array(data['center'])
    keys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'age', 'smass', 'feh', 'popid']
    useful_data = []
    for key in keys:
        useful_data += [data[key]]
    useful_data = np.array(useful_data).T
    df = pd.DataFrame(useful_data, columns=keys)

    # Convert age to gigayears
    df['age'] = 10**df['age'] / 10**9

    df = calculate_lifetimes(df)

    # Make data centred on galactic centre
    df.loc[:, ['px', 'py', 'pz', 'vx', 'vy', 'vz']] += centre

    # Add column which specifies remnant type
    rtypes = np.array(['']*len(df), dtype='<U20')
    smasses = df['smass'].to_numpy()
    rtypes[25 <= smasses] = 'Black Hole'
    rtypes[(8 <= smasses) & (smasses < 25)] = 'Neutron Star'
    rtypes[smasses < 8] = 'White Dwarf'
    df['rtype'] = rtypes
    
    # Make all remnants invisible by default
    df['gmag'] = np.inf

    return df

def add_masses(df, masses):
    '''Add mass to each entry of the DataFrame'''
    for i in df.index.values:
        df.loc[i, 'mass'] = masses.get_mass(df.loc[i, 'rtype'], df.loc[i, 'smass'])
    return df

def add_kicks(df, natal_kicks, logger=None, verbose=0):
    '''Add kick to each entry of the DataFrame'''
    if logger is None:
        verbose = 0
    else:
        logger = logger.getChild(__name__)
    
    for prog, i in enumerate(df.index.values):
        if verbose and prog % (df.shape[0]//100) == 0:
            logger.info(f'Creating kicks, progress = {100 * prog / df.shape[0]:.0f}%')

        vx, vy, vz = natal_kicks.get_kick(df.loc[i, 'rtype'], df.loc[i, 'mass'], df.loc[i, 'smass'])

        df.loc[i, ['vx', 'vy', 'vz']] += np.array([vx, vy, vz])

    return df

def update_cylindrical_coords(df):
    '''Updates the cylindrical coordinates based on values of the cartesian coordinates'''
    x = df['px']
    y = df['py']
    vx = df['vx']
    vy = df['vy']
    df['R'] = np.sqrt(x**2 + y**2)
    df['phi'] = np.arctan2(y, x)
    df['vR'] = (x*vx + y*vy)/df['R']
    df['vphi'] = (x*vy - y*vx)/df['R']**2
    df['vT'] = df['R']*df['vphi']
    return df

def update_cartestian_coordinates(df):
    '''Updates the cartestian coordinates based on values of the cylindrical coordinates'''
    R = df['R']
    phi = df['phi']
    vR = df['vR']
    vT = df['vT']
    df['vphi'] = vT/R
    vphi = df['vphi']
    df['px'] = R * np.cos(phi)
    df['py'] = R * np.sin(phi)
    df['vx'] = vR*np.cos(phi) - R*vphi*np.sin(phi)
    df['vy'] = vR*np.sin(phi) + R*vphi*np.cos(phi)
    return df

def get_final_locations(df, timesteps):
    '''Retrieves the final location of the remnant from galpy output'''
    ages = np.array(df['age'] - df['lifetime']).reshape(-1, 1)
    timesteps = timesteps.reshape(1, -1)
    timesteps = np.repeat(timesteps, ages.shape[0], axis=0)

    # Find the argument where the remnant should have finished orbiting
    final_args = np.argmin(np.abs(timesteps.to(u.Gyr).value - ages), axis=1)
    return final_args

def calculate_orbits(df, duration=None, logger=None, verbose=0):
    """
    Calculates the orbit of each remnant in the provided DataFrame.

    By default the orbits are calculated for a duration based on the age of the
    star but with the lifetime of the original star subtracted (so they are
    evolved for the duration of the remnants life). If a duration is specified
    then all remnants are evolved for this period of time (value is assumed to
    be in Gyr).

    Parameters
    ---------
    df : DataFrame
        pandas DataFrame containing the information on the remnants being evolved
    duration : int (optional)
        The duration (in Gyr) for which to calculate orbits

    Returns
    ----------
    DataFrame
        pandas DataFrame containing the updated remnants with their evolved
        positions
    """
    
    if logger is None:
        verbose = 0
    else:
        logger = logger.getChild(__name__)
    ro, vo = 8.0, 232.0
    remnant_starts = np.array(df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']])
    conversion_to_natural_units = np.array([ro, vo, vo, ro, vo, 1])
    remnant_starts /= conversion_to_natural_units
    # units = [u.kpc, u.km/u.s, u.km/u.s, u.kpc, u.km/u.s, u.radian]

    if duration is not None:
        if duration < 1e-3:
            # Ensure that for small durations there are at least 10 steps
            step = duration / 10
            timesteps = np.arange(0, duration + step, step)*u.Gyr
        else:
            # Else the step size is set to 100,000 years
            timesteps = np.arange(0, duration + 1e-4, 1e-4)*u.Gyr
    else:
        # Step size is set to 1 million years
        timesteps = np.arange(0, df['age'].max() + 1e-3, 1e-3)*u.Gyr
    orbit_values = []
    step = 512
    complete = 0

    for i in range(0, remnant_starts.shape[0], step):
        if i >= complete / 100 * remnant_starts.shape[0] and verbose >= 1:
            complete = (100 * i) // (remnant_starts.shape[0])
            logger.info(f'Orbits are {complete}% complete.')
            complete += 1
        # new_orbits = Orbit(remnant_starts[i:min(i+step, remnant_starts.shape[0])], ro=8.0*u.kpc, vo=232.0 * u.km/u.s)
        new_orbits = Orbit(remnant_starts[i:min(i+step, remnant_starts.shape[0])])
        new_orbits.integrate(timesteps, MWPotential2014, method='symplec6_c')
        orbit_values.append(new_orbits.getOrbit())

    if verbose >= 1:
        logger.info('Finished calculating orbits')
    orbit_values = np.vstack(orbit_values)
    orbit_values *= conversion_to_natural_units
    if verbose >= 1:
        logger.info('Finished vstacking')
    
    if duration is not None:
        df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']] = orbit_values[:, -1]
    else:
        final_orbits = (np.arange(df.shape[0]), get_final_locations(df, timesteps))
        if verbose >= 1:
            logger.info(f'{orbit_values.shape}, Max age: {df["age"].max()}')
        df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']] = orbit_values[final_orbits]
    return update_cartestian_coordinates(df)

def tag_escaping(df):
    '''Calculates whether the remnant will escape the galaxy'''
    df['will_escape'] = np.sqrt(np.sum(df[['vx', 'vy', 'vz']]**2, axis=1)) \
        >= potential.vesc(MWPotential2014, 
                          np.sqrt(np.sum(df[['px', 'py', 'pz']]**2, axis=1))/8.0)*232
    return df
