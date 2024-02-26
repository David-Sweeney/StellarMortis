import warnings
import os
import pickle
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u

from ..utils import get_logger

# Stop ErfaWarnings
warnings.filterwarnings('ignore', module='erfa')

def make_plot_dir(dir_path):
    """Creates a folder to store plots in if it doesn't already exist.

    Args:
    
        dir_path (str): Path to the folder to create
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return

def calculate_yearly_stats(table, bins):
    """
    Calculates yearly statistics for the given table.
    
    Args:
        table (Astropy.QTable): Table containing the data
        bins (dict): Dictionary containing the bins for each plot type
        undersample (float): Factor by which the data is undersampled
    
    Returns:
        dict: Dictionary containing the yearly statistics
    """
    undersample = table.undersample
    
    start_time = Time(table.data.meta['Start time'], format='isot')
    end_time = start_time + TimeDelta(sum(table.data.meta['Years of observation'])*365.25*u.day)
    
    # Remove a year either side to avoid edge effects
    start_time = start_time + TimeDelta(365.25*u.day)
    end_time = end_time - TimeDelta(365.25*u.day)
    
    timesteps = np.arange(start_time.jd, end_time.jd+1e-3, 365.25)
        
    # Reduce table to only contain events
    temp_table = table.events.copy()
    temp_table.sort('event time')

    # Find yearly indices in the table
    temp_times = temp_table['event time'].jd
    event_inds = np.searchsorted(temp_times, v=timesteps, side='left')

    # Prepare output dictionary
    stats = {}
    stats['num > 1 mas'] = []
    stats['num > 2 mas'] = []
    stats['num > 5 mas'] = []
    stats['blended num > 1 mas'] = []
    stats['blended num > 2 mas'] = []
    stats['blended num > 5 mas'] = []
    stats['num > 2 mag'] = []
    stats['num > 10 mag'] = []
    stats['num > 1 bmag'] = []
    stats['num > 2.5 bmag'] = []
    stats['Shifts'] = []
    stats['Blended Shifts'] = []
    stats['Magnifications'] = []
    stats['Bump Magnitudes'] = []
    stats['us'] = []
    stats['Separations'] = []
    
    # For each year, calculate the statistics
    for i in range(len(event_inds)-1):
        start = event_inds[i]
        end = event_inds[i+1]
        year_table = temp_table[start:end]
        
        # Astrometric shifts
        stats['num > 1 mas'].append(sum(year_table['centroid shift'] > 1*u.mas)*undersample)
        stats['num > 2 mas'].append(sum(year_table['centroid shift'] > 2*u.mas)*undersample)
        stats['num > 5 mas'].append(sum(year_table['centroid shift'] > 5*u.mas)*undersample)
        
        stats['blended num > 1 mas'].append(sum(year_table['blended centroid shift'] > 1*u.mas)*undersample)
        stats['blended num > 2 mas'].append(sum(year_table['blended centroid shift'] > 2*u.mas)*undersample)
        stats['blended num > 5 mas'].append(sum(year_table['blended centroid shift'] > 5*u.mas)*undersample)
        
        # Photometric magnifications
        stats['num > 2 mag'].append(sum(year_table['lensing magnification'] > 1)*undersample)
        stats['num > 10 mag'].append(sum(year_table['lensing magnification'] > 9)*undersample)
        
        stats['num > 1 bmag'].append(sum(year_table['bump magnitude'] > 1*u.mag)*undersample)
        stats['num > 2.5 bmag'].append(sum(year_table['bump magnitude'] > 2.5*u.mag)*undersample)
        
        # Histograms
        values, _ = np.histogram(year_table['centroid shift'].to(u.mas).value, bins=bins['Shifts'],
                                 weights=np.ones_like(year_table['centroid shift'])*undersample)
        stats['Shifts'].append(values)
        
        values, _ = np.histogram(year_table['blended centroid shift'].to(u.mas).value, bins=bins['Blended Shifts'],
                                 weights=np.ones_like(year_table['blended centroid shift'])*undersample)
        stats['Blended Shifts'].append(values)
        
        values, _ = np.histogram(year_table['lensing magnification'], bins=bins['Magnifications'],
                                 weights=np.ones_like(year_table['lensing magnification'])*undersample)
        stats['Magnifications'].append(values)
        
        values, _ = np.histogram(year_table['bump magnitude'].to(u.mag).value, bins=bins['Bump Magnitudes'],
                                 weights=np.ones_like(year_table['bump magnitude'])*undersample)
        stats['Bump Magnitudes'].append(values)
                                 
        us = (year_table['min separation']/year_table['einstein angle']).decompose().value
        values, _ = np.histogram(us, bins=bins['us'],
                                 weights=np.ones_like(year_table['min separation'])*undersample)
        stats['us'].append(values)
                                 
        values, _ = np.histogram(year_table['min separation'].to(u.mas).value, bins=bins['Separations'],
                                 weights=np.ones_like(year_table['min separation'])*undersample)
        stats['Separations'].append(values)
    
    # Convert all yearly stats to numpy arrays
    for key, value in stats.items():
        stats[key] = np.array(value)
    
    return stats

def bootstrap_confidence_interval(tables, output_dir, bootstraps=1e3):
    """Calculates the 95% confidence interval for various quantities.
    
    Quantities which need a confidence interval:
        Number of events > 1 mas
        Number of events > 10 mas
        Number of events > 1 magnification
        Number of events > 9 magnification
        >= astrometric shift
        >= magnification
        <= u_0
        <= theta_0
        
    Saves the results as a dictionary in a pickle file in the specified folder. The 
    dictionary is of the form:
        {species: {statistic: (lower CI, upper CI)}}
        
        Where the species strings are:
            'BHs'
            'NSs'
            'Stars'
            
        The bin and bin centres are also stored in the dictionary and can be accessed:
            {'bins'/'bin centres': {plot: values}}
        
            Where plot is one of: 'Shifts', 'Magnifications', 'Separations', 'us'
        
        And the statistic strings are:
            'num > 1 mas'
            'num > 10 mas'
            'num > 1 mag'
            'num > 9 mag'
            'Shifts'
            'Magnifications'
            'us'
            'Separations'
        
        The Shifts, Magnifications, us and Separations statistics are 1D numpy arrays
        containing lower/upper CI values for each of the bins. The other statistics are 
        simply the lower/upper CI.

    Args:
        guw_table (Astropy.QTable): Table containing the GUW data
        folder_name (str): Name of the folder in which to save the uncertainty information
        mw_table (Astropy.QTable, optional): Table containing the stellar data. Defaults to None.
        bootstraps (int, optional): Number of bootstraps to perform. Defaults to 10.
    """
    
    bootstraps = int(bootstraps)
    
    
    # Calculate the bins for each plot type
    bin_dict = {}
    bin_centres_dict = {}
    for plot in ['Shifts', 'Blended Shifts', 'Magnifications', 'Bump Magnitudes', 
                 'Separations', 'us']:
        if plot == 'Shifts':
            data = [table.events['centroid shift'].to(u.mas).value for table in tables]
        elif plot == 'Blended Shifts':
            data = [table.events['blended centroid shift'].to(u.mas).value for table in tables]
        elif plot == 'Magnifications':
            data = [table.events['lensing magnification'].value for table in tables]
        elif plot == 'Bump Magnitudes':
            data = [table.events['bump magnitude'].to(u.mag).value for table in tables]
        elif plot == 'Separations':
            data = [table.events['min separation'].to(u.mas).value for table in tables]
        elif plot == 'us':
            data = [(table.events['min separation']/table.events['einstein angle']).decompose().value for table in tables]
        else:
            raise ValueError(f'Unknown plot type: {plot}.')
        
        # Calculate most extreme data point
        extreme_value = None
        extreme_data = None
        for i in range(len(data)):
            if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                        'Bump Magnitudes', ]:
                value = np.amax(data[i])
                if extreme_value is None or value > extreme_value:
                    extreme_value = value
                    extreme_data = data[i]
            elif plot in ['Separations', 'us']:
                value = np.amin(data[i])
                if extreme_value is None or value < extreme_value:
                    extreme_value = value
                    extreme_data = data[i]
            else:
                raise ValueError(f'Unknown plot type: {plot}.')

        # Get bins
        bins, bin_centres = get_log_bins(extreme_data, 100)
        
        # Save bins into dictionary
        bin_dict[plot] = bins
        bin_centres_dict[plot] = bin_centres
        
    # Prepare dictionary to store uncertainty statistics
    all_species_CIs = {}
    
    # Calculate confidence intervals for each species
    for table in tables:
        all_species_CIs[table.species] = {}
        
        # Get each stat measured over a year
        yearly_stats = calculate_yearly_stats(table, bin_dict)
        num_years = len(yearly_stats['num > 1 mas'])
        
        # Select the indices for each bootstrap
        np.random.seed(0)
        inds = np.random.randint(0, num_years, size=(bootstraps, num_years))
        
        # Calculate the statistics for each bootstrap
        bootstrapped_data = {key: [] for key in yearly_stats.keys()}
        for i in range(bootstraps):
            for key in yearly_stats.keys():
                bootstrapped_data[key].append(np.sum(yearly_stats[key][inds[i]], axis=0))
        
        # Compute the confidence intervals based on the bootstraps
        for key, value in bootstrapped_data.items():
            lower_CI = np.percentile(bootstrapped_data[key], 2.5, axis=0)
            upper_CI = np.percentile(bootstrapped_data[key], 97.5, axis=0)
            all_species_CIs[table.species][key] = (lower_CI, upper_CI)   
        
    # Add in bins and bin_centres to the dictionary
    all_species_CIs['bins'] = bin_dict
    all_species_CIs['bin centres'] = bin_centres_dict
    
    # Save to file
    with open(f'{output_dir}/uncertainty_stats.pkl', 'wb') as f:
        pickle.dump(all_species_CIs, f)
        
    return

def save_summary_stats(tables, output_dir):
    """Save summary statistics"""
    def write_out_data(f, text, table_col_name, unit, uncertainty_col_name):
        total_events = 0
        for table in tables:
            events = table.events
            total_events += len(events[events[table_col_name] >= unit])*table.undersample
        f.write(f'{text}: {total_events:n}\n')
        
        for table in tables:
            events = table.events
            num = len(events[events[table_col_name] >= unit])*table.undersample
            ci = uncertainty_stats[table.species][uncertainty_col_name]
            f.write(f'  {table.species}: {num:n} (- {num-ci[0]:.3f}, + {ci[1]-num:.3f})\n')
        
        f.write('\n')
        
        return

    # Get uncertainties
    with open(f'{output_dir}/uncertainty_stats.pkl', 'rb') as f:
        uncertainty_stats = pickle.load(f)

    # Some statistics for the paper
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write(f"{'='*10 + ' '*5} OTHER EVENTS {' '*5 + '='*10}\n")
        # Astrometric shifts
        write_out_data(f, 'Number of events which have a shift >= 1 mas', 'centroid shift', 1*u.mas, 'num > 1 mas')
        write_out_data(f, 'Number of events which have a shift >= 2 mas', 'centroid shift', 2*u.mas, 'num > 2 mas')
        write_out_data(f, 'Number of events which have a shift >= 5 mas', 'centroid shift', 5*u.mas, 'num > 5 mas')
        
        # Blended shifts
        write_out_data(f, 'Number of events which have a blended shift >= 1 mas', 'blended centroid shift', 1*u.mas, 'blended num > 1 mas')
        write_out_data(f, 'Number of events which have a blended shift >= 2 mas', 'blended centroid shift', 2*u.mas, 'blended num > 2 mas')
        write_out_data(f, 'Number of events which have a blended shift >= 5 mas', 'blended centroid shift', 5*u.mas, 'blended num > 5 mas')
        
        # Photometric magnifications
        write_out_data(f, 'Number of events which have a magnification >= 2', 'lensing magnification', 1, 'num > 2 mag')
        write_out_data(f, 'Number of events which have a magnification >= 10', 'lensing magnification', 9, 'num > 10 mag')
        
        # Bump magnitudes
        write_out_data(f, 'Number of events which have a bump magnitude >= 1', 'bump magnitude', 1*u.mag, 'num > 1 bmag')
        write_out_data(f, 'Number of events which have a bump magnitude >= 2.5', 'bump magnitude', 2.5*u.mag, 'num > 2.5 bmag')
        f.write('\n')
            
    return

def get_bin_centres(bins):
    return [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

def get_log_bins(quantity, num_bins, min_quantity=None, max_quantity=None):
    """Create logarithmically spaced bins of the given quantity"""
    if min_quantity is None:
        min_quantity = min(quantity)
        if min_quantity < 2e-16:
            warnings.warn('Minimum quantity is less than 2e-16, setting to 2e-16.')
            min_quantity = 2e-16
    if max_quantity is None:
        max_quantity = max(quantity)
    min_bin = np.log10(min_quantity)
    max_bin = np.log10(max_quantity)
    bins = np.logspace(min_bin, max_bin, num_bins+1)

    bin_centres = get_bin_centres(bins)

    # Increase first and last bin slightly to make sure to include extreme points
    bins[0] = bins[0]*(1 - 1e-5)
    bins[-1] = bins[-1]*(1 + 1e-5)

    return bins, bin_centres

def plot(tables, output_dir, plot, bright_threshold=0*u.mag, logger=None, verbose=1):
    """
    Plots the given tables in a manner specified by the plot parameter.
    
    Parameters
    ----------
    tables : list of MicrolensingTable
        List of tables to plot
    output_dir : str
        Directory in which to save the plots
    plot : str
        Type of plot to make. Options are:
            'Shifts' : Plot the unblended astrometric shifts
            'Blended Shifts' : Plot the blended astrometric shifts
            'Magnifications' : Plot the unblended photometric magnifications
            'Bump Magnitudes' : Plot the (blended) bump magnitudes
            'Separations' : Plot the angular separations
            'us' : Plot the dimensionless separations
            'Einstein times' : Plot the distribution Einstein times, both in number and as a fraction of the total
            'Einstein angles' : Plot the distribution Einstein angles, both in number and as a fraction of the total
            'Event locations' : Plot the distribution of events in Galactic coordinates
            'Event sky locations' : Plot the distribution of events in the sky
            'Time-Parallax scatter' : Plot the Einstein time-microlens parallax scatter
            'Lens distance' : Plot the distribution of lens distances
            'Lens mass' : Plot the distribution of lens masses
    bright_threshold : astropy magnitude, optional
        Bump magnitude threshold for the plot. Aside from the distribution plots (i.e. Shifts, Blended Shifts, 
        Magnifications, Bump Magnitudes, Separations, us), all plots will only include events with bump magnitudes 
        smaller than this threshold. Defaults to 0 mag (i.e. all events).
    """

    for table in tables:
        table.set_bright(bright_threshold)
    
    # Plots BH shifts and magnification
    if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 'Bump Magnitudes', 'Separations', 'us']:
        plt.figure()
        
        if plot == 'Shifts':
            data = [table.events['centroid shift'].to(u.mas).value for table in tables]
            bound = 1e-2
            title = 'Astrometric Shifts'
            x_label = 'Centroid shift (mas)'
            filename = 'Shifts'
            cumulative = -1
        elif plot == 'Blended Shifts':
            data = [table.events['blended centroid shift'].to(u.mas).value for table in tables]
            bound = 1e-2
            title = 'Blended Astrometric Shifts'
            x_label = 'Blended centroid shift (mas)'
            filename = 'Blended_Shifts'
            cumulative = -1
        elif plot == 'Magnifications':
            data = [table.events['lensing magnification'].value for table in tables]
            bound = 1e-6
            title = 'Photometric Magnifications'
            x_label = 'Lensing magnification - 1'
            filename = 'Magnifications'
            cumulative = -1
        elif plot == 'Bump Magnitudes':
            data = [table.events['bump magnitude'].to(u.mag).value for table in tables]
            bound = 1e-4
            title = 'Bump Magnitudes'
            x_label = 'Bump magnitude (mag)'
            filename = 'Bump_Magnitudes'
            cumulative = -1
        elif plot == 'Separations':
            data = [table.events['min separation'].to(u.mas).value for table in tables]
            bound = 1e3
            title = 'Microlensing Angular Separations'
            x_label = r'$\theta_0$ (mas)'
            filename = 'Separations'
            cumulative = True
        elif plot == 'us':
            data = [(table.events['min separation']/table.events['einstein angle']).decompose().value for table in tables]
            bound = 1e2
            title = 'Microlensing Dimensionless Separations'
            x_label = '$u_0$'
            filename = 'us'
            cumulative = True
        
        # Set bins
        extreme_value = None
        extreme_data = None
        # Calculate most extreme data point
        for i in range(len(data)):
            if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                        'Bump Magnitudes']:
                value = np.amax(data[i])
                if extreme_value is None or value > extreme_value:
                    extreme_value = value
                    extreme_data = data[i]
            elif plot in ['Separations', 'us']:
                value = np.amin(data[i])
                if extreme_value is None or value < extreme_value:
                    extreme_value = value
                    extreme_data = data[i]
        
        # I should just set bounds ***
        bins, bin_centres = get_log_bins(extreme_data, 100)
        smallest_bin_index = sum(bins < bound)
        
        # Load uncertainty data
        with open(f'{output_dir}/uncertainty_stats.pkl', 'rb') as f:
            uncertainty_dict = pickle.load(f)
        
        assert np.all(uncertainty_dict['bins'][plot] == bins), 'Bins do not match'
        assert np.all(uncertainty_dict['bin centres'][plot] == bin_centres), 'Bin centres do not match'
        
        if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                    'Bump Magnitudes']:
            bins = bins[smallest_bin_index - 1:]
            bin_centres = bin_centres[smallest_bin_index - 1:]
            xs = np.concatenate((bin_centres, bins[[-1]]))
        elif plot in ['Separations', 'us']:
            bins = bins[:smallest_bin_index + 1]
            bin_centres = bin_centres[:smallest_bin_index]
            xs = np.concatenate((bins[[0]], bin_centres))
        
        # Plot data and uncertainties
        for table in tables:
            if plot == 'Shifts':
                data = table.events['centroid shift'].to(u.mas).value
            elif plot == 'Blended Shifts':
                data = table.events['blended centroid shift'].to(u.mas).value
            elif plot == 'Magnifications':
                data = table.events['lensing magnification'].value
            elif plot == 'Bump Magnitudes':
                data = table.events['bump magnitude'].to(u.mag).value
            elif plot == 'Separations':
                data = table.events['min separation'].to(u.mas).value
            elif plot == 'us':
                data = (table.events['min separation']/table.events['einstein angle']).decompose().value
            else:
                raise ValueError(f'Invalid plot: {plot}')
            
            plt.hist(data, bins=bins, weights=table.undersample*np.ones_like(data), 
                        cumulative=cumulative, histtype='step', label=table.species, color=table.colour)
            
            
            # Calculate uncertainty points
            lower_CI, upper_CI = uncertainty_dict[table.species][plot]
            
            if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                        'Bump Magnitudes']:
                # Turn confidence intervals into reverse cumulative distributions
                lower_CI = np.cumsum(lower_CI[::-1])[::-1]
                upper_CI = np.cumsum(upper_CI[::-1])[::-1]
                
                lower_CI = lower_CI[smallest_bin_index - 1:]
                upper_CI = upper_CI[smallest_bin_index - 1:]
            elif plot in ['Separations', 'us']:
                # Turn confidence intervals into reverse cumulative distributions
                lower_CI = np.cumsum(lower_CI)
                upper_CI = np.cumsum(upper_CI)
                
                lower_CI = lower_CI[:smallest_bin_index]
                upper_CI = upper_CI[:smallest_bin_index]
            else:
                raise ValueError(f'Invalid plot: {plot}')
            
            
            # Ensure confidence interval doesn't drop below the undersampling
            upper_CI = np.clip(upper_CI, table.undersample, None)
            if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                        'Bump Magnitudes']:
                lower_CI = np.hstack((lower_CI, lower_CI[[-1]]))
                upper_CI = np.hstack((upper_CI, upper_CI[[-1]]))
            elif plot in ['Separations', 'us']:
                lower_CI = np.hstack((lower_CI[[0]], lower_CI))
                upper_CI = np.hstack((upper_CI[[0]], upper_CI))
            else:
                raise ValueError(f'Invalid plot: {plot}')
            
            
            # Plot uncertainty boundary 
            plt.fill_between(xs, lower_CI, upper_CI, alpha=0.1, color=table.colour)
            plt.plot(xs, lower_CI, '--', alpha=0.25, color=table.colour)
            plt.plot(xs, upper_CI, '--', alpha=0.25, color=table.colour)
            
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(x_label)
        plt.ylabel('Cumulative events (yr$^{-1}$)')
        plt.title(title)
        
        if plot in ['Shifts', 'Blended Shifts', 'Magnifications', 
                    'Bump Magnitudes']:
            plt.xlim(left=bound)
        elif plot in ['Separations', 'us']:
            plt.xlim(right=bound)
        else:
            raise ValueError(f'Invalid plot: {plot}')
        
        min_undersample = min([table.undersample for table in tables])
        bottom = min(0.5*min_undersample, 1)
        plt.ylim(bottom=bottom)
        plt.legend()
        
        plt.tight_layout()

        plt.savefig(f'{output_dir}/{filename}.png')
        plt.close()
    elif plot in ['Einstein times', 'Einstein angles']:
        if plot == 'Einstein times':
            col = 'einstein time'
            unit = u.day
            
            filename1 = 'Time_number_distribution'
            filename2 = 'Time_fraction_distribution'
            title1 = r'Distribution of events by $t_E$'
            title2 = r'Event break-down by $t_E$'
            xlabel = '$t_E$ (day)'
            num_bins = 20
        elif plot == 'Einstein angles':
            col = 'einstein angle'
            unit = u.mas
            
            filename1 = 'Angle_number_distribution'
            filename2 = 'Angle_fraction_distribution'
            title1 = r'Distribution of events by $\theta_E$'
            title2 = r'Event break-down by $\theta_E$'
            xlabel = r'$\theta_E$ (mas)'
            num_bins = 30
        else:
            raise ValueError(f'Invalid plot: {plot}')    
            
        # Add bright threshold to title and filename
        if bright_threshold != 0:
            title1 += f' for $\\Delta m$ < -{bright_threshold:.0e}'
            title2 += f' for $\\Delta m$ < -{bright_threshold:.0e}'
            filename1 += f'_{bright_threshold:.0e}'
            filename2 += f'_{bright_threshold:.0e}'
        
        plt.figure()
        all_events = np.concatenate([table.bright[col].to(unit).value for table in tables])
        all_event_weights = np.concatenate([table.undersample*np.ones_like(table.bright[col]) for table in tables])
        bins, bin_centres = get_log_bins(all_events, num_bins, min_quantity=all_events.min(), 
                                         max_quantity=all_events.max())
        
        plt.hist(all_events, bins=bins, color='k', histtype='step', lw=1.75,
                                    weights=all_event_weights, label='All')
        for table in tables:
            plt.hist(table.bright[col].to(unit).value, bins=bins, lw=1.75, ls=table.linestyle,
                     histtype='step', color=table.colour, 
                     weights=table.undersample*np.ones_like(table.bright[col]), label=table.species)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel('Number (yr$^{-1}$)')
        plt.title(title1)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{filename1}.png')
        plt.close()
        
        # Create fractional plot
        plt.figure()
        
        all_values, _ = np.histogram(all_events, bins=bins, weights=all_event_weights)
        for table in tables:
            values, _ = np.histogram(table.bright[col].to(unit).value, bins=bins, 
                                     weights=table.undersample*np.ones_like(table.bright[col]))
            plt.plot(bin_centres, values/all_values, marker='s', linewidth=1, label=table.species, c=table.colour)
        
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel('Fractional contribution')
        plt.title(title2)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{filename2}.png')
        plt.close()
    elif plot in ['Event locations']:

        plt.figure()
        all_longs = []
        all_weights = []
        for table in tables:
            sc = SkyCoord(ra=table.bright['bgs.ra'], dec=table.bright['bgs.dec'],
                          frame='icrs').transform_to(Galactic)
            
            longs = sc.l.to(u.deg).value
            # Recentre longitudes in [-180, 180] instead of [0, 360]
            longs[longs > 180] -= 360
            
            all_longs.append(longs)
            all_weights.append(table.undersample * np.ones_like(longs))
                
            sns.kdeplot(longs, bw_adjust=1, color=table.colour, label=table.species)
        
        sns.kdeplot(np.hstack(all_longs), bw_adjust=1, weights=all_weights, color='k', zorder=1.5, label='All')

        plt.xlabel('Galactic Longitude (deg)')
        plt.ylabel('Density')
        plt.title('Distribtuion of Microlensing Events in Galactic Longitude')
        plt.legend()
        plt.xlim([-100, 100])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/Glon_{bright_threshold:.0e}.png')
        plt.close()
        
        plt.figure()
        all_lats = []
        all_weights = []
        for table in tables:
            sc = SkyCoord(ra=table.bright['bgs.ra'], dec=table.bright['bgs.dec'],
                          frame='icrs').transform_to(Galactic)
            
            lats = sc.b.to(u.deg).value
            
            all_lats.append(lats)
            all_weights.append(table.undersample * np.ones_like(lats))
                
            sns.kdeplot(lats, bw_adjust=1, color=table.colour, label=table.species)
        
        sns.kdeplot(np.hstack(all_lats), bw_adjust=1, color='k', zorder=1.5, label='All')
        
        plt.xlabel('Galactic Latitude (deg)')
        plt.ylabel('Density')
        plt.title('Distribtuion of Microlensing Events in Galactic Latitude')
        plt.legend()
        plt.xlim([-20, 20])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/Glat_{bright_threshold:.0e}.png')
        plt.close()
    elif plot in ['Event sky locations']:        
        plt.figure()
        plt.subplot(111, projection='mollweide')
        plt.grid(True)
        
        all_longs = []
        all_lats = []
        for table in tables:
            sc = SkyCoord(ra=table.bright['bgs.ra'], dec=table.bright['bgs.dec'],
                          frame='icrs').transform_to(Galactic)
            longs = sc.l.to(u.deg).value
            longs[longs > 180] -= 360
            lats = sc.b.to(u.deg).value
            
            all_longs.append(longs)
            all_lats.append(lats)
            
            plt.scatter(np.radians(longs), np.radians(lats), s=5, alpha=0.2, color=table.colour, label=table.species)
        
        plt.title('Microlensing events in Galactic coordinates')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Event_sky_locations_{bright_threshold:.0e}.png')
        plt.close()
        
        df = pd.DataFrame()
        df['Species'] = np.hstack([[table.species]*len(table.bright) for table in tables])
        df['RA'] = np.hstack([table.bright['bgs.ra'].to(u.deg).value for table in tables])
        df['Dec'] = np.hstack([table.bright['bgs.dec'].to(u.deg).value for table in tables])
        df['l'] = np.hstack(all_longs)
        df['b'] = np.hstack(all_lats)
        df.to_csv(f'{output_dir}/Event_sky_locations_{bright_threshold:.0e}.csv')
    elif plot in ['Time-Parallax scatter']:
        plt.figure()
        
        for table in tables:
            lens_parallax = table.bright['lens.distance'].to(u.mas, equivalencies=u.parallax())
            # pi_E = pi_rel / theta_E
            microlens_parallax = np.abs(lens_parallax - table.bright['bgs.parallax'])/table.bright['einstein angle']
            plt.scatter(table.bright['einstein time'].to(u.day).value, microlens_parallax.decompose(), 
                        color=table.colour, alpha=0.5, label=table.species)
        
        plt.title('Einstein time vs. Einstein parallax')
        plt.xlabel('$t_E$ (days)')
        plt.ylabel('$|\\pi_E|$')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/Time-Parallax_scatter_{bright_threshold:.0e}.png')
        plt.close()
    elif plot in ['Lens distance']:
        plt.figure()
        for table in tables:
            plt.hist(table.bright['lens.distance'].to(u.kpc).value, bins=100, histtype='step', 
                     weights=table.undersample*np.ones_like(table.bright['lens.distance']), 
                     ls = table.linestyle, color=table.colour, lw=1.75, density=True, 
                     range=(0, 16), label=table.species)

        plt.xlabel('Distance to Lens (kpc)')
        plt.ylabel('Event density')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Lens_distance.png')
        plt.close()
    elif plot in ['Lens mass']:
        for table in tables:
            plt.hist(table.bright['lens.mass'].to(u.Msun).value, bins=35, histtype='step', 
                     weights=table.undersample*np.ones_like(table.bright['lens.mass']), 
                     color=table.colour, lw=1.75, density=True, label=table.species)
        plt.xlabel('Mass of lens ($M_\odot$)')
        plt.ylabel('Probability density')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Lens_mass.png')
        plt.close()
    else:
        raise ValueError(f'Unkown plot type: {plot}')   
    
    if verbose >= 1 and logger is not None:
        if bright_threshold == 0*u.mag:
            logger.info(f'Created {plot}')
        else:
            logger.info(f'Created {bright_threshold:1.0e} {plot}')
    return
            
def plot_all(tables, output_dir, bootstraps=1000, save_summary=True, logger=None, verbose=1):
    """
    Plot a wide range of plots for the given tables.

    Parameters
    ----------
    tables : list of MicrolensingTables
        List of MicrolensingTable objects to plot
    output_dir : str
        Directory in which to save the plots
    bootstraps : int or float, optional
        Number of bootstraps to perform. Set bootstraps to None to reuse a previous 
        bootstrap file in output_dir.  If a float is provided it is converted to an int. 
        The default is 1000.
    save_summary : bool, optional
        Whether to save the summary statistics to a file. The default is True.
    """
    
    if logger is None:
        logger = get_logger()
    else:
        logger = logger.getChild(__name__)
    
    make_plot_dir(output_dir)
    
    if bootstraps is not None:
        if verbose >= 1:
            start_time = time.time()
            logger.info('Starting bootstrap')
        bootstrap_confidence_interval(tables, output_dir, bootstraps=bootstraps)
        if verbose >= 1:
            logger.info(f'Finished bootstrap in {(time.time() - start_time)/60**2:.1f} hours')
    
    if save_summary:
        # Summarise everything
        save_summary_stats(tables, output_dir)
    
    # Make plots for distribution plots
    for type in ['Shifts', 'Blended Shifts', 'Magnifications', 'Bump Magnitudes', 'Separations', 
                 'us', 'Einstein times', 'Einstein angles', 'Lens distance']:
        plot(tables, output_dir, plot=type, logger=logger, verbose=verbose)
    
    plot(tables, output_dir, plot='Time-Parallax scatter', bright_threshold=1.75*u.mag, logger=logger, verbose=verbose)
    plot(tables, output_dir, plot='Einstein times', bright_threshold=1e-10*u.mag, logger=logger, verbose=verbose)
    plot(tables, output_dir, plot='Einstein angles', bright_threshold=1e-10*u.mag, logger=logger, verbose=verbose)
    plot(tables, output_dir, plot='Lens mass', bright_threshold=1e-2*u.mag, logger=logger, verbose=verbose)
    plot(tables, output_dir, plot='Event sky locations', bright_threshold=1e-1*u.mag, logger=logger, verbose=verbose)
    
    bright_threshold = 4e-1*u.mag
    for type in ['Einstein times', 'Einstein angles', 'Event locations', 
                 'Time-Parallax scatter']:
        plot(tables, output_dir, plot=type, bright_threshold=bright_threshold, logger=logger, verbose=verbose)
    