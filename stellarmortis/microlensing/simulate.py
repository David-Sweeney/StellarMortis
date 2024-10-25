import warnings
import logging
import os
import time
import shutil
from socket import gethostname

import numpy as np
import pandas as pd
import ebf
from requests.exceptions import RequestException
import astropy.coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy import constants
from astropy.table import QTable, vstack
from astropy.utils.metadata import MergeConflictWarning
from astroquery.gaia import Gaia
try:
    import ray
except ModuleNotFoundError:
    warnings.warn('Ray not installed, cannot simulate microlensing events')

from ..utils import get_logger

logging.getLogger('astroquery').setLevel(logging.WARNING)
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Select Data Release 3
Gaia.ROW_LIMIT = -1

# Stop ErfaWarnings
warnings.filterwarnings('ignore', module='erfa')
    
def load_csv(filepath, start=0, end=None):
    
    # Read in column names from file
    columns = pd.read_csv(filepath, nrows=0).columns
    
    # Set number of rows to read
    if end is None:
        nrows = None
    else:
        nrows = end - start
        
    # Skip to correct section of df (add 1 to start since the first row is the header)
    df = pd.read_csv(filepath, names=columns, skiprows=start+1, nrows=nrows)
    
    return df

def load_ebf(filepath, start=0, end=None):
    
    data = ebf.read(filepath, '/')
    coords = [data['px'], data['py'], data['pz'], data['vx'], data['vy'], data['vz']]
    centre = data['center']
    coords = np.array(coords).T + np.array(centre[:6])
    
    df = pd.DataFrame(coords, columns=['px', 'py', 'pz', 'vx', 'vy', 'vz'])
    df['rtype'] = 'Star'
    df['R'] = np.linalg.norm(df[['px', 'py']], axis=1)
    df['mass'] = data['mact']
    
    distance_from_earth = np.linalg.norm([data['px'], data['py'], data['pz']], axis=0)
    
    # Conversion into apparent magnitudes from Galaxia documentation using CTIO V and R filters
    apparent_vs = data['ubv_v'] + np.log(100*distance_from_earth) + data['exbv_schlegel']*3.240
    apparent_rs = data['ubv_r'] + np.log(100*distance_from_earth) + data['exbv_schlegel']*2.634
    
    # Compute Gaia G magnitudes using Johnson-Cousins relationships specified in Gaia documentation
    v_minus_r = apparent_vs - apparent_rs
    g_mag = apparent_vs + -0.03088 + -0.04653*v_minus_r + -0.8794*v_minus_r**2 + 0.1733*v_minus_r**3
    df['gmag'] = g_mag
    
    if end is not None:
        df = df.iloc[start:end]
    else:
        df = df.iloc[start:]
    
    df = df.reset_index(drop=True)
    return df
    
def load_df_for_skycoords(filepath, start_time, start=0, end=None):
    if filepath.endswith('.csv'):
        df = load_csv(filepath, start, end)
    elif filepath.endswith('.ebf'):
        df = load_ebf(filepath, start, end)
    else:
        raise ValueError('File must be in .csv or .ebf format')
    
    if not set(('ra', 'dec', 'distance', 'pm_ra_cosdec', 'pm_dec', 'radial_velocity')).issubset(df.columns):
    
        sc = SkyCoord(frame='galactocentric',
                    x=df['px']*u.kpc, y=df['py']*u.kpc,
                    z=df['pz']*u.kpc, v_x=df['vx']*(u.km/u.s),
                    v_y=df['vy']*(u.km/u.s), v_z=df['vz']*(u.km/u.s),
                    obstime=start_time).transform_to(coords.builtin_frames.ICRS)
        
        df['ra'] = sc.ra.to(u.deg).value
        df['dec'] = sc.dec.to(u.deg).value
        df['distance'] = sc.distance.to(u.kpc).value
        df['pm_ra_cosdec'] = sc.pm_ra_cosdec.to(u.mas/u.yr).value
        df['pm_dec'] = sc.pm_dec.to(u.mas/u.yr).value
        df['radial_velocity'] = sc.radial_velocity.to(u.km/u.s).value
    
    assert set(('ra', 'dec', 'distance', 'pm_ra_cosdec', 'pm_dec', 'radial_velocity')).issubset(df.columns), 'SkyCoords not loaded correctly'
    assert set(('rtype', 'mass', 'gmag')).issubset(df.columns), 'Missing columns in data file'
    
    return df

def get_file_length(filepath):
    if filepath.endswith('.ebf'):
        data = ebf.read(filepath, '/')
        return len(data['px'])
    with open(filepath, 'r') as f:
        for count, line in enumerate(f):
            pass
    return count + 1

def calculate_max_shift_separations():
    """
    Calculate the dimensionless separation with the maximum blended centroid 
    shifts for various lens/source flux ratios.

    Returns:
        1D numpy array: Array containing the lens/source flux ratios
        1D numpy array: Array containing the dimensionless separation with 
            the maximum blended centroid shifts
    """
    
    us = np.logspace(-3, 0.151, 20000).reshape(1, -1)
    gs = np.logspace(-5, 6, 1000).reshape(-1, 1)
    cs = 1/(1+gs) * ((1 + gs*(us**2 - us*np.sqrt(us**2 + 4) + 3))/(us**2 + 2 + gs*us*np.sqrt(us**2 + 4)))*us
    maxs = us[0, np.argmax(cs, axis=1)]
    
    return gs.squeeze(), maxs.squeeze()

def approximate_buffer_window(sensitivity, lens_mass, lens_parallax):
    """
    Approximate buffer around lens path within which a background source must
    fall to achieve a specified deviation in apparent position.

    This function uses the approximation from Kluter et al. (2022). The default
    deviation is 0.1 mas. The buffer is increased by the size of the lens
    parallax to ensure that potential pairs are not missed.

    Parameters
    ----------
    lens : SkyCoord
        Astropy SkyCoord object of the lens
    lens_mass : Quantity (mass)
        Astropy Quantity of the mass of the lensing object
    sensitivity : Quantity (angle)
        Astropy Quantity of the size of astrometric shift that defines the
        region

    Returns
    -------
    Quantity (angle)
        An Astropy Quantity of the required angular separation (in arcseconds)
    """

    # 95% of background stars were found to have a parallax larger than this value
    bgs_parallax = -1.2 * u.mas

    # Calculate Einstein angle in mas
    theta_E = ((np.sqrt((4*constants.G)/(constants.c**2 * u.pc))
                * np.sqrt(lens_mass * (lens_parallax - bgs_parallax)/u.arcsec)
                ).decompose() * u.rad).to(u.mas)

    # Approximation from Kluter et al. (2022)
    # If sensitivity is given in angle units
    if sensitivity.unit.is_equivalent(u.mas):
        approximation = (theta_E**2/(sensitivity)).to(u.arcsec)
    elif sensitivity.unit.is_equivalent(u.mag):
        # If sensitivity is given in magnitude units
        raise NotImplementedError('Sensitivity in magnitude units not yet implemented')
        # Make approximation numerically?
    else:
        raise ValueError(f'Sensitivity must be in angle or magnitude units, not {sensitivity.unit}')

    # Add lens parallax
    approximation = approximation + lens_parallax + abs(bgs_parallax)

    return approximation

def get_earth(ts):
    """
    Get the position of the earth in barycentric coordinates.
    
    The position is calculated using an ellipse which I
    fitted from the 2013 positions of the Astropy get_sun() 
    function. The ellipse repeats every year.
    
    Parameters
    ----------
    ts : numpy array
        Array containing the times in days to get the Earth's position
    
    Returns
    -------
    numpy array
        Array containing the x, y, z coordinates of the earth in AU.
        Array shape is (3, T) where T is the number of times provided.
    """

    # Convert ts to radians
    ts = ts.reshape(-1, 1)*2*np.pi/365.25
    
    # Predefined orbital parameters
    f0 = np.array([0.00559436, -0.02239685, -0.00970921])
    f1 = np.array([-0.18424293, 0.90164277, 0.39087722])
    f2 = np.array([-0.98273009, -0.16903882, -0.07328005])
    
    positions = f0 + f1*np.cos(ts) + f2*np.sin(ts)
    
    return positions.T

def move_point(p, t, parallax=None):
    """
    Move a point according to its proper motion
    
    Parameters
    ----------
    p : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the point
    t : float
        Time in days
    parallax : float (optional)
        Parallax of the point in mas. If specified then the point
        will also be moved according to its parallax.
    
    Returns
    -------
    numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the point after motion
    """
    
    # Convert t to years
    t = t/365.25
    
    # Define declinations in degrees
    dec = p[1] + p[3]/(1000*60*60)*t
    
    # Define right ascensions in degrees
    ra = p[0] + p[2]/(1000*60*60)*t
    
     # Calculate parallax if specified
    if parallax is not None:
        
        # Convert parallax from mas to radians
        parallax = np.radians(parallax/(1000*60*60))
        
        # Convert RA and Dec to radians
        ra = np.radians(ra)
        dec = np.radians(dec)
        
        # Get Earth positions
        x, y, z = get_earth(np.array(t)*365.25)
        
        # This parallax correction comes from the Kovalevsky & Seidelmann (2004) textbook
        d_dec = parallax*((x*np.cos(ra)+y*np.sin(ra))*np.sin(dec) - z*np.cos(dec))
        d_ra = parallax/np.cos(dec) * (x*np.sin(ra)-y*np.cos(ra))
        
        # Apply parallax
        dec = dec - d_dec
        ra = ra - d_ra
        
        # Convert RA and Dec back to degrees
        ra = np.degrees(ra)
        dec = np.degrees(dec)
    
    # Check if motion has gone over the poles (+- 90 deg)
    if abs(dec) > 90:
        assert abs(dec) < 180
        # Pole correction
        correction = np.clip(2*(abs(dec) - 90), 0, None)
        dec = dec - np.sign(dec)*correction
        ra = (ra + 180*(correction != 0)) % 360
    
    new = np.array((ra, dec, p[2], p[3]))
    
    return new

def haversine(t, p1, p2, parallax1=None, parallax2=None):
    """
    Calculates the haversine distance between two points.
    
    Parameters
    ----------
    t : float or numpy array of floats
        Time in days. If parallax is provided then this is the
        time since the start of 2013 for the purposes of calculating
        parallax
    p1 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the first point
    p2 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the second point
    parallax1 : float (optional)
        Parallax of the first point in mas. If one parallax is 
        provided then they both must be.
    parallax2 : float (optional)
        Parallax of the second point in mas. If one parallax is 
        provided then they both must be.
    
    Returns
    -------
    float
        Haversine distance between the two points in mas
    """
    
    # Convert t to years
    t = t/365.25
    
    # Define declinations in radians
    dec_1 = np.radians(p1[1] + p1[3]/(60*60*1000)*t)
    dec_2 = np.radians(p2[1] + p2[3]/(60*60*1000)*t)
    
    # Define right ascensions in radians
    ra_1 = np.radians(p1[0] + p1[2]/(60*60*1000)*t)
    ra_2 = np.radians(p2[0] + p2[2]/(60*60*1000)*t)
    
    # Calculate parallax if specified
    if parallax1 is not None:
        assert (parallax1 is not None) and (parallax2 is not None), 'Parallaxes must be provided for both points'
        
        # Convert parallax from mas to radians
        parallax1 = np.radians(parallax1/(1000*60*60))
        parallax2 = np.radians(parallax2/(1000*60*60))
        
        # Get Earth positions
        x, y, z = get_earth(np.array(t)*365.25)
        
        # This parallax correction comes from the Kovalevsky & Seidelmann (2004) textbook
        d_dec_1 = parallax1*((x*np.cos(ra_1)+y*np.sin(ra_1))*np.sin(dec_1) - z*np.cos(dec_1))
        d_ra_1 = parallax1/np.cos(dec_1) * (x*np.sin(ra_1)-y*np.cos(ra_1))
        d_dec_2 = parallax2*((x*np.cos(ra_2)+y*np.sin(ra_2))*np.sin(dec_2) - z*np.cos(dec_2))
        d_ra_2 = parallax2/np.cos(dec_2) * (x*np.sin(ra_2)-y*np.cos(ra_2))
        
        # Apply parallax
        dec_1 = dec_1 - d_dec_1
        ra_1 = ra_1 - d_ra_1
        dec_2 = dec_2 - d_dec_2
        ra_2 = ra_2 - d_ra_2
    
    # Correct coordinates if motion has gone over the poles (+- 90 deg)
    if np.amax((np.abs(dec_1), np.abs(dec_2))) > np.pi/2:
        # Pole correction for p1
        assert np.abs(dec_1).max() < np.pi
        correction = np.clip(2*(np.abs(dec_1) - np.pi/2), 0, None)
        dec_1 = dec_1 - np.sign(dec_1)*correction
        ra_1 = (ra_1 + np.pi*(correction != 0)) % (2*np.pi)
        
        # Pole correction for p2
        assert np.abs(dec_2).max() < np.pi
        correction = np.clip(2*(np.abs(dec_2) - np.pi/2), 0, None)
        dec_2 = dec_2 - np.sign(dec_2)*correction
        ra_2 = (ra_2 + np.pi*(correction != 0)) % (2*np.pi)
    
    # Define differences
    d_dec = dec_1 - dec_2
    d_ra = ra_1 - ra_2
    
    inner = np.sin(d_dec/2)**2 + (1 - np.sin(d_dec/2)**2 - np.sin((dec_1+dec_2)/2)**2) * np.sin(d_ra/2)**2
    distance = 2*np.arcsin(np.sqrt(inner))
    
    # Return the distance in mas
    return np.degrees(distance)*60*60*1000

def get_background_stars(lens, years_of_observation, sensitivity, lens_mass, lens_parallax, logger, retry=True):
    """
    Searches Gaia DR3 for possible nearby background stars.

    In future this should be updated to filter out low-quality stars.

    Parameters
    ----------
    lens : SkyCoord
        Astropy SkyCoord object of the lens
    lens_mass : Quantity (mass)
        Astropy Quantity of the mass of the lensing object
    plot : bool (optional)
        Whether to plot the found objects
    times : Time (optional)
        Astropy Time object containing the times over which to plot the
        lens. Required if plot == True
    retry : bool (optional)
        Whether to retry to search the Gaia catalog if the connection is reset

    Returns
    -------
    Table
        An Astropy Table containing background stars which are close to the path
        of the lens
    """

    
    # Half the total duration in days
    half_duration = sum(years_of_observation)*365.25/2

    # Get middle of lens's path as centre for window
    # mid_point = lens.apply_space_motion(dt=half_duration)
    mid_point = move_point(lens, half_duration)
    
    # Buffer for the radius of lensing and lens parallax
    buffer = approximate_buffer_window(sensitivity, lens_mass, lens_parallax)
    
    # Add in a buffer to catch stars which are moving at <10.5 mas/yr
    movement_buffer = sum(years_of_observation)*u.yr*(10.5*u.mas/u.yr)

    radius = haversine(0, mid_point, lens)*u.mas + buffer + movement_buffer
    mid_coord = SkyCoord(ra=mid_point[0], dec=mid_point[1], unit=u.deg, frame='icrs')

    if retry:
        # Loop until connection is successful
        while True:
            try:
                j = Gaia.cone_search_async(coordinate=mid_coord, radius=radius)
                r = j.get_results()
            except ConnectionResetError as e:
                logger.exception('Caught Connection Error!')
                time.sleep(1)
                continue
            except TimeoutError as e:
                logger.exception('Caught Timeout Error!')
                time.sleep(1)
                continue
            except RequestException as e:
                logger.exception('Caught Request Error!')
                time.sleep(300)
                continue
            except OSError:
                logger.exception('Caught OSError!')
                time.sleep(60)
                continue
            except Exception as e:
                logger.exception('Caught unknown error, continuing')
                time.sleep(60)
                continue
            break
    else:
        j = Gaia.cone_search_async(coordinate=mid_point, radius=radius)
        # Get results and fill masked values with zeros
        r = j.get_results()
    
    # Fill null proper motions with approximate median values
    r['pmra'].fill_value = -1.885932688110818
    r['pmdec'].fill_value = -3.974576412703027
    
    # Fill other null values with 0s
    r['ra'].fill_value = 0
    r['dec'].fill_value = 0
    r['parallax'].fill_value = 0
    r['radial_velocity'].fill_value = 0
    r = r.filled()
    
    # Set minimum parallax value
    r['parallax'] = np.clip(r['parallax'], 0.0625*u.mas, None)

    return r

def filter_background_stars(lens_parallax, background_stars):
    """
    Filter out background stars which do not match the criteria.

    The criteria removes background stars which:
        * Are closer than the lens
        * Have a significantly negative parallax

    Parameters
    ----------
    lens : SkyCoord
        Astropy SkyCoord object of the lens
    background_stars : Table
        An Astropy Table containing background stars

    Returns
    -------
    Table
        An Astropy Table containing background stars which meet the specified
        criteria
    """

    background_stars = background_stars[background_stars['parallax'] < lens_parallax]
    background_stars = background_stars[background_stars['parallax']
                                        + 5*background_stars['parallax_error'] > 0]
    
    return background_stars

def minimise(fun, args, bounds, tol=1):
    """
    Minimises a function using a variant of the binary search 
    algorithm.
    
    Parameters
    ----------
    fun : function
        Function to minimise
    args : tuple
        Arguments to pass to the function
    bounds : tuple
        Bounds to search between
    tol : float
        Tolerance of the minimum argument
    
    Returns
    -------
    float
        Minimum value of the function
    float
        Value of the argument at the minimum value
    """
    
    
    steps = 500
    low, high = bounds
    
    # Check if the minimum is at the bounds
    if fun(low, *args) < fun(low+1, *args):
        return fun(low, *args), low
    elif fun(high, *args) < fun(high-1, *args):
        return fun(high, *args), high

    # Two ended binary search
    while high - low > tol:
        xs = np.linspace(low, high, steps)
        vals = fun(xs, *args)
        best_ind = np.argmin(vals)
        low, high = xs[[max(0, best_ind-1), min(steps-1, best_ind+1)]]
        
    return np.amin(vals), xs[np.argmin(vals)]

def get_minimum_separation(p1, p2, initial_time, years_of_observation):
    """
    Finds the minimum separation between two points over the course of the 
    observation period. Parallax is not taken into account.
    
    Parameters
    ----------
    p1 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the first point
    p2 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the second point
    initial_time : Time
        Astropy Time object containing the start time
    years_of_observation : list
        List containing the number of years of observation and the gaps
        between observations. Even indices will be treated as years of
        observation, odd indices will be treated as years of no observation.
    
    Returns
    -------
    Astropy distance
        Minimum separation between the two points in mas
    Time
        Astropy Time object containing the time of the minimum separation
        
    """
    
    end = sum(years_of_observation)*365.25
    min_separation, closest_time = minimise(haversine, (p1, p2), (0.0, end), tol=1)
    
    assert closest_time >= 0, f'Time is before observation period: {closest_time}'
    assert closest_time <= sum(years_of_observation)*365.25, f'Time is after observation period: {closest_time}'
    
    closest_time = initial_time + TimeDelta(closest_time*u.day)
    
    return min_separation*u.mas, closest_time

def check_parallax(p1, parallax1, p2, parallax2, initial_time, years_of_observation, closest_time):
    """
    Rechecks the minimum separation between two points accounting for
    their parallactic motions. 
    
    If there is only one observation period, this is done by checking 
    the separation between the two points every day for a year before 
    and after the reported closest time.
    
    If there are multiple observation periods this function checks every 
    day in the observation periods
    
    Parameters
    ----------
    p1 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the first point
    parallax1 : float
        Parallax of the first point in mas
    p2 : numpy array
        Array containing the ra (in degrees), dec (in degrees), 
        pm_ra (in mas/yr), pm_dec (in mas/yr) of the second point
    parallax2 : float
        Parallax of the second point in mas
    initial_time : Time
        Astropy Time object containing the start time
    years_of_observation : list
        List containing the number of years of observation and the gaps
        between observations. Even indices will be treated as years of
        observation, odd indices will be treated as years of no observation.
    closest_time : Time
        Astropy Time object containing the time of the minimum separation
        before parallax is taken into account
    
    Returns
    -------
    Astropy distance
        Minimum separation between the two points in mas
    Time
        Astropy Time object containing the time of the minimum separation
    """
    
    # Convert closest time to days since observation began
    closest_time = closest_time.jd - initial_time.jd
    
    # If there are multiple observation periods then check every day
    if len(years_of_observation) > 1:
        # Stop side effects from occurring
        years_of_observation = years_of_observation.copy()
        years_of_observation.insert(0, 0)
        cumulative_years = np.cumsum(years_of_observation)*365.25
        times = []
        for i in range(0, len(years_of_observation), 2):
            times.append(np.arange(cumulative_years[i], 
                                   cumulative_years[i+1], 1))
        times = np.hstack(times)
        
        # Calculate the separation between the two points each day
        dists = haversine(times, p1, p2, parallax1.to(u.mas).value, 
                          parallax2.to(u.mas).value)
        closest_time = times[np.argmin(dists)]
        
        # Work out which observation period the closest time is in
        index = np.searchsorted(cumulative_years, closest_time)
        if (index % 2) == 0:
            # If the closest time the start of an observation boundary, place the index inside that boundary
            index += 1
        observing_bounds = (cumulative_years[index-1], cumulative_years[index])
        
    # Otherwise check either side of the closest time
    else:
        # The number of years before and after the closest time to check
        observation_window = 1e2
        observing_bounds = (0, sum(years_of_observation)*365.25)
        while True:
            # Get each day for a year before and after the closest time, respecting the observation bounds
            earliest_time = max(closest_time-observation_window*365.25, 0)
            latest_time = min(closest_time+observation_window*365.25, 
                              sum(years_of_observation)*365.25)
            times = np.arange(earliest_time, latest_time, 1)
        
            # Calculate the separation between the two points each day
            dists = haversine(times, p1, p2, parallax1.to(u.mas).value, 
                            parallax2.to(u.mas).value)
            closest_time = times[np.argmin(dists)]
            
            # If the closest time is near the edge of the observation period then increase the observation window
            if closest_time < times[len(times)//100] and earliest_time != 0:
                observation_window *= 2
                continue
            elif (closest_time > times[-len(times)//100] 
                  and latest_time != sum(years_of_observation)*365.25):
                observation_window *= 2
                continue
            
            # Otherwise the closest time has been found
            break
            
    # Recalculate the separation at a finer level
    # Ensure the recalculation is within the observation bounds
    earliest_time = max(closest_time-1, observing_bounds[0])
    latest_time = min(closest_time+1, observing_bounds[1])
    times = np.arange(earliest_time, latest_time, 0.01)
    dists = haversine(times, p1, p2, parallax1.to(u.mas).value, 
                      parallax2.to(u.mas).value)
    
    # Convert closest time back into a date
    closest_time = initial_time + TimeDelta(times[np.argmin(dists)]*u.day)
    
    return np.amin(dists)*u.mas, closest_time

def collate_files(progress_filepaths, progress_dir, output_filepath, run_name, delete_progress=True, delete_task_dir=True, 
                  combine_dates=False, big_events_only=False, logger=None):
    """Combine QTables from list of filenames to one file"""

    if logger is None:
        logger = get_logger()
    logger.info('Collating files')
    
    tables = []
    metadata = None
    dates = []
    
    progress_filenames = [os.path.basename(filepath) for filepath in progress_filepaths]
    
    while not set(progress_filenames) <= set(os.listdir(progress_dir)):
        logger.info(f'Missing files, waiting 30 minutes: {set(progress_filenames) - set(os.listdir(progress_dir))}')
        time.sleep(30*60)
        
    for filepath in progress_filepaths:
        logger.info(f'Collating {filepath}')
        try:
            table = QTable.read(filepath)
        except FileNotFoundError as e:
            logger.exception(f'FileNotFoundError for {filepath} in collate_files.')
            raise e
        assert (metadata is None 
                or table.meta['Years of observation'] == metadata['Years of observation']), \
            f'Years of observation do not match for {filepath}'
        if big_events_only:
            table = table[table['blended centroid shift'] >= 1e-2*u.mas]
            # table = table[table['lensing event'].astype(bool)]
        tables.append(table)

        if metadata is None:
            metadata = table.meta
        if combine_dates:
            dates.append(table.meta['Start time'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MergeConflictWarning)
        # Exact join so an error is thrown if there is an issue
        combined_table = vstack(tables, join_type='exact')
    combined_table.meta = metadata
    if combine_dates:
        combined_table.meta['Start time'] = dates

    logger.info(f'Writing out data to {output_filepath}')
    combined_table.write(output_filepath, overwrite=True)

    # Remove used files
    if delete_progress:
        for filepath in progress_filepaths:
            os.remove(filepath)
    
    if delete_task_dir:
        # Remove task files
        shutil.rmtree(f'{progress_dir}/tasks/{run_name}')
        
        # Remove task directory if empty (i.e. no other runs are using it)
        if len(os.listdir(f'{progress_dir}/tasks')) == 0:
            os.rmdir(f'{progress_dir}/tasks')

    return

@ray.remote(num_cpus=0.2)
def main(filepath, output_dir, years_of_observation, sensitivity, logging_file, start=0, end=None, run_name='', verbose=0):
    """
    Runs the lensing calculation pipeline from loading in the data to saving the
    data out.

    The start and end index of the DataFrame can optionally be specified to
    allow for parallelisation. If either one is not specified that value
    defaults to the start/end of the DataFrame. Files are saved out with
    start-end appended to the filepath.

    Parameters
    ---------
    start : int
        The index to start with in the DataFrame
    end : int
        The index to go up to in the DataFrame (exclusive)
    species : str
        The lens population to use. The options are:
            'underworld' -  black holes and neutron stars
            'milkyway' - galaxia model of the visible stars
    offset : int
        Offset in years for data to be loaded (default is None, which is 0)
    verbose : int
        Specifies how much should be outputted to logging file. Large values
        also report information from smaller values:
            0 - Start, progress, saving, end
            1 - Information on found events
            2 - Major steps in function

    Returns
    ----------
    None
    """
    try:
        
        logger = get_logger(logging_file, True)
        
        logging.getLogger('astroquery').setLevel(logging.WARNING)
        warnings.filterwarnings('ignore', module='erfa')
        
        assert end is not None, 'End must be specified'
        assert start < end, 'Start must be less than end'
        # assert filepath.endswith('.csv'), 'Filepath must be an .csv file'

        if verbose >= 3:
            logger.info(f'Starting main() with: start={start}, end={end}, verbose={verbose}')
        
        # Create the Astropy Time object used throughout the process
        start_time = Time('2013-01-01T00:00:0')
        
        # Load lens data
        df = load_df_for_skycoords(filepath, start_time, start=start, end=end)
        
        # Initialise ouput lists
        output_data = []
        output_data_columns = ['lens.ra', 'lens.dec', 'lens.distance',
                               'lens.pmra', 'lens.pmdec',
                               'lens.radial_velocity', 'lens.g_mag', 'lens.mass',
                               'lens.species', 
                               'lensing event', 'bgs.gaia_source_id', 'bgs.ra', 
                               'bgs.dec', 'bgs.pmra', 'bgs.pmdec', 
                               'bgs.radial_velocity', 'bgs.g_mag', 'bgs.parallax',
                               'min separation', 'einstein angle', 'einstein time',
                               'major image shift', 'centroid shift', 
                               'blended centroid shift', 'lensing magnification', 
                               'bump magnitude', 'event time']
        
        # Null values which will be used if no bgs is sufficiently close
        null_bgs_values = [0, 0, coords.Longitude(0*u.deg),
                           coords.Latitude(0*u.deg),
                           0*(u.mas/u.yr), 0*(u.mas/u.yr), 0*(u.km/u.s), 0*u.mag, 0*u.mas,
                           0*u.arcsec, 0*u.mas, 0*u.day, coords.Angle(0*u.mas), 
                           coords.Angle(0*u.mas), coords.Angle(0*u.mas), 
                           0*u.dimensionless_unscaled, -0*u.mag, 
                           Time(0, format='jd')]

        # Calculate maximum photocentre shift separations
        gs, maxs = calculate_max_shift_separations()
        
        if verbose >= 3:
            logger.info(f'{start}-{end} starting incidence calculations')

        # Set logging interval
        if verbose >= 2:
            logs = end - start
        elif verbose == 1:
            logs = 10
        else:
            logs = 4
        # Ensure logging interval can't be 0, which would cause an error
        logging_interval = max((end - start)//logs, 1)

        for i in range(len(df)):
            if (i % logging_interval == 0) and (verbose >= 1):
                logger.info(f'{start}-{end} reached index {i}...')
            
            lens = np.array((df.iloc[i]['ra'], df.iloc[i]['dec'], 
                             df.iloc[i]['pm_ra_cosdec'],
                             df.iloc[i]['pm_dec']))
            
            lens_mass = df.iloc[i]['mass'] * u.M_sun
            lens_distance = df.iloc[i]['distance']*u.kpc
            lens_parallax = lens_distance.to(u.arcsec, u.parallax())
            lens_mag = df.iloc[i]['gmag']*u.mag
            lens_species = df.iloc[i]['rtype']
            lens_info = [coords.Longitude(lens[0]*u.deg), coords.Latitude(lens[1]*u.deg), 
                         lens_distance.to(u.pc), lens[2]*(u.mas/u.yr),
                         lens[3]*(u.mas/u.yr), df.iloc[i]['radial_velocity']*(u.km/u.s), 
                         lens_mag, lens_mass, lens_species]

            # Get sky around test point
            if verbose >= 3:
                logger.info('Getting background stars')
            objects = get_background_stars(lens, years_of_observation, sensitivity, 
                                           lens_mass=lens_mass, lens_parallax=lens_parallax,
                                           logger=logger, retry=True)

            # Filter objects
            if verbose >= 3:
                logger.info('Filtering background stars')
            objects = filter_background_stars(lens_parallax, objects)

            if len(objects) == 0:
                event_info = lens_info
                event_info.extend(null_bgs_values)
                output_data.append(event_info)
                if verbose >= 2:
                    logger.info(f'No sources found for {i} in range {start}-{end}')
                continue
            else:
                lens_info.append(True)
                event_info = lens_info.copy()

            # Test each object for microlensing
            if verbose >= 3:
                logger.info('Checking each object')
            event_infos = []
            for object in objects:
                event_info = lens_info.copy()
                bgs_parallax = object['parallax']*u.mas
                bgs_mag = object['phot_g_mean_mag']*u.mag
            
                bgs = np.array((object['ra'], object['dec'], 
                                 object['pmra'], object['pmdec']))

                if verbose >= 3:
                    logger.info('Calculating minimum separation')
                
                min_separation, closest_time = get_minimum_separation(lens, bgs, start_time, 
                                                                      years_of_observation)
                
                # Throw out events which are too far away
                theta_E = ((np.sqrt((4*constants.G)/(constants.c**2 * u.pc))
                                * np.sqrt(lens_mass * (lens_parallax - bgs_parallax)/u.arcsec)
                                ).decompose() * u.rad)
                approx_shift = abs(theta_E**2/(min_separation - abs(lens_parallax - bgs_parallax)))
                if (approx_shift < sensitivity*0.8
                    and min_separation - abs(lens_parallax - bgs_parallax) > 0):
                    continue
                
                if verbose >= 3:
                    logger.info('Recalculating lens-BGS distance with parallax taken into account')
                # Recalculate minimum separation with parallax taken into account
                # Parallax is increased by 1% as Gaia orbits L2
                min_separation, closest_time = check_parallax(lens, lens_parallax*1.01, bgs, bgs_parallax*1.01,
                                                              start_time, years_of_observation, closest_time)
                
                event_info.extend([object['SOURCE_ID'], coords.Longitude(bgs[0]*u.deg), 
                                   coords.Latitude(bgs[1]*u.deg),
                                   bgs[2]*(u.mas/u.yr),  bgs[3]*(u.mas/u.yr),
                                   object['radial_velocity']*(u.km/u.s), bgs_mag, 
                                   bgs_parallax, min_separation])
                
                if verbose >= 3:
                    logger.info('Calculating lensing event')

                # Calculate magnitude of lensing event
                u_ = min_separation/theta_E
                
                relative_speed = np.sqrt((lens[2]*(u.mas/u.yr) 
                                          - bgs[2]*(u.mas/u.yr))**2 
                                         + (lens[3]*(u.mas/u.yr) - bgs[3]*(u.mas/u.yr))**2)
                
                einstein_time = (theta_E/relative_speed).to(u.day)
                
                # Calculate the photometric magnification
                magnification = (u_**2 + 2)/(u_*np.sqrt(u_**2 + 4)) - 1
                magnification = np.clip(magnification, 0, None) # Floating point errors can cause magnification to be negative
                
                # Calculate blended bump magnification
                g = (100**(1/5))**(bgs_mag - lens_mag).to(u.mag).value
                bump_mag = 2.5*np.log10((1 + magnification + g)/(1+g)) * u.mag
                
                # Calculate the astrometric shift
                major_image_shift = 0.5*(-u_ + np.sqrt(u_**2 + 4))*theta_E
                
                # Calculate maximum photocentre shift
                clipped_u = np.clip(u_, np.sqrt(2), None)
                photocentre_shift = theta_E * clipped_u / (clipped_u**2 + 2)
                
                # Calculate the maximum photocentre shift accounting for blending with a luminous lens
                u_max = np.interp(g, gs.squeeze(), maxs.squeeze())
                u_cLL = max(u_, u_max)
                blended_shift = theta_E/(1+g) * ((1 + g*(u_cLL**2 - u_cLL*np.sqrt(u_cLL**2 + 4) + 3))
                                                 /(u_cLL**2 + 2 + g*u_cLL*np.sqrt(u_cLL**2 + 4)))*u_cLL

                if major_image_shift < sensitivity:
                    continue
                else:
                    event_info.extend([theta_E, einstein_time, major_image_shift,
                                       photocentre_shift, blended_shift, magnification, 
                                       bump_mag, closest_time])
                event_infos.append(event_info)

            if verbose >= 3:
                logger.info('Appending output')
            # Add appropriate output to output_data
            if len(event_infos) == 0:
                # If no event occurred append null info
                # Remove the flag for an event having ocurred
                event_info = lens_info[:-1]
                event_info.extend(null_bgs_values)
                output_data.append(event_info)
                
                if verbose >= 2:
                    logger.info(f'No sources large for {i} in range {start}-{end}')
            else:
                # If an event ocurred update with number of events
                number_of_events = len(event_infos)

                # Whether an event flag is at the end of lens_info
                for event_info in event_infos:
                    event_info[len(lens_info) - 1] = number_of_events

                output_data.extend(event_infos)
                if verbose >= 2:
                    logger.info(f'Found {number_of_events} for {i} in range {start}-{end}')

        # Transpose output lists
        output_data = list(map(list, zip(*output_data)))
        
        # Save out data
        if verbose >= 1:
            logger.info(f'{start}-{end} saving...')
        out_qt = QTable(data=output_data, names=output_data_columns,
                        meta={'Start time': str(start_time),
                              'Years of observation': years_of_observation,
                              'Corrected major image shift': True,
                              'Gaia table': "gaiadr3.gaia_source",
                              'GUW file': filepath})
        
        out_qt['event time'].format = 'jd'
        out_qt['einstein angle'] = out_qt['einstein angle'].to(u.mas)
        out_qt['centroid shift'] = out_qt['centroid shift'].to(u.mas)
        
        # Check that events occur after start time
        events = out_qt[out_qt['lensing event'].astype(bool)]
        assert np.all(events['event time'] >= start_time), f'Event time before start time: {np.amin(events["event time"])}'
        # Check that events occur before end time
        assert np.all(events['event time'] 
                      <= start_time + TimeDelta(sum(years_of_observation)*365.25*u.day)), 'Event time after end time'
        
        out_qt.write(f'{output_dir}/{run_name}_{start}-{end}.ecsv', overwrite=True)
            
        if verbose >= 1:
            logger.info(f'{start}-{end} now finished!')

    except Exception as e:
        print('Exception!')
        if 'i' not in locals():
            i = 'undefined'
        logger.exception(f'Exception from range {start}-{end} at index: {i}')

    return 1

def parallelised_main(filepath, progress_dir, years, sensitivity, run_name='', start=0, end=None, num_workers=1, logger=None, verbose=0):
    """
    Runs main in parallel using Ray.

    This function manages how many processes are active and will kill a process
    if it takes too long. This is definied as >10 seconds per source, typically
    remnants will take 4 seconds. Killed processes will be added to the end of
    the queue so that they can be rerun. This handling is done as there is some
    rare error which causes a process to never terminate (or crash but not inform
    the main process).

    Args:
        offset (str, optional): Offset in years for data to be loaded. Defaults to None.
        species (str, optional): The lens population to use. Defaults to 'underworld'. 
            The options are:
                'underworld' -  black holes and neutron stars
                'milkyway' - galaxia model of the visible stars
        start (int, optional): Starting index of the data for which to calculate incidence. 
            Defaults to 0.
        end (int, optional): Final index of the data for which to calculate incidence.
            Defaults to None which means that the final index is set to be the length of the data
        verbose (int, optional): Specifies how much information should be outputted to 
            logging file. Defaults to 0. Large values also report information from smaller 
            values:
                0 - Start, progress, saving, end
                1 - Information on found events
                2 - Major steps in function
        

    Returns:
        int: Step size used when writing out to files.
    """
    
    # Parallelisation can't cope with passing the logger directly. Instead, the
    # logging file is passed and a new logger is created in each process.
    if isinstance(logger.handlers[0], logging.FileHandler):
        logging_file = logger.handlers[0].baseFilename
    else:
        logging_file = None
    
    logger = logger.getChild(__name__)
        
    if not isinstance(years, list):
        years = list(years)
        
    if sensitivity is not None:
        # Check that sensitivity is an astropy Quantity with units of angle
        try:
            sensitivity.to(u.mas)
        except Exception:
            raise ValueError('Sensitivity must be an astropy Quantity with units of angle.')
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'File {filepath} not found')
    
    if end is None:
        data_length = get_file_length(filepath) - 1
        end = data_length
        
    # Set step size so batches aren't too fast
    if sum(years[::2]) < 2e3:
        step_size = 500
    elif sum(years[::2]) < 2e4:
        step_size = 250
    else:
        step_size = 10
    
    # Set the step size so that the number of tasks is at least the number of cores
    step_size = min(step_size, int(np.ceil((end - start)/num_workers)))
    
    max_running_tasks = num_workers

    # Set up directory for the unstarted tasks
    directory = f'{progress_dir}/tasks/{run_name}/'
    unstarted_directory = directory + 'unstarted/'
    completed_directory = directory + 'completed/'
    
    if os.path.exists(directory):
        print(f'Run directory "{directory}" already exists.')
        decision = input('What would you like to do? (d)elete or (j)oin: ')
        if decision == 'd':
            print('Deleting...')
            shutil.rmtree(directory)
        elif decision == 'j':
            # Will join the existing run
            print('Joining...')
            warnings.warn(f'Run directory "{directory}" already exists, joining existing run.')
            logger.info(f'Run directory "{directory}" already exists, joining existing run.')
        else:
            raise ValueError(f'Invalid decision: {decision}')
    
    if not os.path.exists(directory):
        os.makedirs(unstarted_directory)
        os.mkdir(completed_directory)
            
        index_range = list(range(start, end, step_size))
        if index_range[-1] != end:
            index_range.append(end)

        number_of_tasks = len(index_range) - 1
        
        # Create file for every task
        task_names = [f'{index_range[i]}-{index_range[i+1]}.task' for i in range(len(index_range)-1)]
        for task in task_names:
            with open(unstarted_directory + task, 'w'):
                pass
    else:
        number_of_tasks = len(os.listdir(unstarted_directory))

    ray.init(ignore_reinit_error=True)
    
    current_tasks = {} # Dictionary indexed by object containing (time, (start, end)) tuples
    task_running_times = []

    # Allowed runtime is set to 5*expected runtime (which is itself overestimated)
        # Gaia query time is ~3.6 seconds
        # Calculation time is the other two terms
    # This scales linearly with the number of sources examined (step_size)
    expected_runtime = (3.6 + 5e-4*sum(years[::2]) 
                        + 2.8e-8*sum(years[::2])**2)*step_size
    allowed_runtime = 20*expected_runtime
    expected_total_runtime = expected_runtime * number_of_tasks/max_running_tasks * u.s

    # Format the runtime string appropriately
    if expected_total_runtime > 1*u.day:
        runtime_str = f'{expected_total_runtime.to(u.day).value:.1f} days'
    elif expected_total_runtime > 1*u.hr:
        runtime_str = f'{expected_total_runtime.to(u.hr).value:.1f} hours'
    else:
        runtime_str = f'{expected_total_runtime.to(u.min).value:.0f} minutes'
     
    if verbose >= 1:   
        logger.info(f'Parallelisation starting')
        logger.info(f'Using {max_running_tasks} workers for an expected runtime of {runtime_str}')
        logger.info(f'A total of {end-start} objects must be processed.')
        logger.info(f'Each block of {step_size} objects is expected to take {(expected_runtime*u.s).to(u.hr).value:.1f} hours.')
    start_time = time.time()

    while len(os.listdir(unstarted_directory)) > 0 or len(current_tasks) > 0:
        # Start tasks with Ray only when they can actually start
        while (len(current_tasks) < max_running_tasks) and (len(os.listdir(unstarted_directory)) > 0):
            # Get new unstarted task
            try:
                # Grab the first task in the directory
                new_task = os.listdir(unstarted_directory)[0]
                os.remove(unstarted_directory + new_task)
                
                # Get task offset, start and end from the filename
                start, end = new_task.split('.')[0].split('-')
                start = int(start)
                end = int(end)
            except FileNotFoundError:
                # Another process has grabbed this task
                logger.info(f'Failed to grab {new_task}')
                time.sleep(5)
                continue
            except IndexError:
                # There are no new tasks to start
                time.sleep(600)
                continue
                
            # Start running new task
            obj = main.remote(filepath, progress_dir, years, sensitivity, logging_file, start=start, end=end, run_name=run_name, verbose=verbose)
            logger.info(f'{gethostname()} is starting: {start}-{end}') # ***

            # Add task info to current_tasks
            current_tasks[obj] = (time.time(), (start, end))

        # Check all tasks for completion
        finished_tasks, remaining_tasks = ray.wait(list(current_tasks.keys()), timeout=1)
        # Print out the logging tasks only when a task finishes
        if len(finished_tasks) != 0:
            running_tasks = [task[1][0] for task in current_tasks.values()]
            logger.info(f'Running tasks: {running_tasks}')

        # Remove completed tasks from current_tasks 
        for task in finished_tasks:
            try:
                assert ray.get(task) == 1, f'Task failed, {current_tasks[task][1]}'
            except ray.exceptions.RayTaskError:
                logger.info(f'Task failed, likely out of memory: {current_tasks[task][1]}')
                
                # Set the task to be treated as if cancelled (i.e. replace in unstarted tasks)
                remaining_tasks.append(task)
                
                # Flag that the task has been killed
                current_tasks[task][0] = None
            task_time, task_seed = current_tasks[task]

            # Create file to show the task has been completed
            with open(completed_directory + f'{task_seed[0]}-{task_seed[1]}.task', 'w'):
                pass
            task_running_times.append(time.time() - task_time)
            logger.info(f'Completed {task_seed} in {((time.time() - task_time)*u.s).to(u.hr).value:.1f} hours')

            # Remove the task from current tasks
            ray.cancel(task)
            del current_tasks[task]

        # Terminate tasks which haven't finished after they have been running for too long
        for task in remaining_tasks:
            task_time, task_seed = current_tasks[task]
            # Tasks are terminated if they exceed the allowed runtime
            if task_time is None or time.time() - task_time > allowed_runtime:
                # Cancel the task
                print(f'Cancelling {task_seed}')
                ray.cancel(task)
                
                if task_time is None:
                    logger.info(f'Handling killed task: {task_seed}')
                else:
                    logger.info(f'Cancelled {task_seed} in because it ran for {(time.time() - task_time)/60/60:.1f} hours')

                # Remove the task from current tasks
                del current_tasks[task]

                # Place the seed back amongst the unstarted tasks
                with open(unstarted_directory + f'{task_seed[0]}-{task_seed[1]}', 'w'):
                    pass
    
    # All tasks have been finished
    if verbose >= 1:
        logger.info(f'All tasks completed in {((time.time() - start_time)*u.s).to(u.hr).value:.1f} hours, exiting.')
        if len(task_running_times) > 0:
            logger.info(f'Min, median, max of task running times: {min(task_running_times):.1f}, '
                        f'{np.median(task_running_times):.1f}, {max(task_running_times):.1f}')
    return step_size
