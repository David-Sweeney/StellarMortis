import logging
import sys

import numpy as np
from astropy.table import QTable
from astropy.time import Time, TimeDelta
import astropy.units as u

class MicrolensingTable():
    def __init__(self, species, filepath, orig_undersample, trim=True):
        """
        Class to store microlensing data for a given species.

        Parameters
        ----------
        species : str
            The species of the lensing object.
        filepath : str
            The filepath to the ecsv file containing the microlensing data.
        orig_undersample : float
            The original undersample factor for the data. This is converted to 
            the factor which the microlensing data undersamples a year of data by
            by dividing by the number of years of observation (accounting for trim
            if trim is True).
        trim : bool, optional
            Whether to trim the data to the first year of observation. Default is True.
        """
        self.species = species
        self.filepath = filepath
        self.data, self.undersample = self._load_table(orig_undersample, trim=trim)
        self.events = self.data[self.data['lensing event'].astype(bool)]
        self.bright = None
        self.bright_threshold = None
        
        # Set the colour that will be used when plotting this population
        self.colour = self._get_colour()
        self.linestyle = self._get_linestyle()
        
    def _load_table(self, orig_undersample, trim):
        """Loads specified table from ecsv file and sets the undersample factor."""
        table = QTable.read(self.filepath)
        # Filter table to only include the species of interest
        table = table[table['lens.species'] == self.species]
        
        years = sum(table.meta['Years of observation'])
        
        if trim:
            # Remove a year either side to avoid edge effects
            assert len(table.meta['Years of observation']) == 1, "Cannot trim multiple periods of data."
            assert years > 2, "Cannot trim less than 2 years of data."
            years = years - 2
            start_time = Time(table.meta['Start time'], format='isot') + TimeDelta(365.25*u.day)
            end_time = start_time + TimeDelta(years*365.25*u.day)
            table = table[(table['event time'] >= start_time) & (table['event time'] <= end_time)]
            
        return table, orig_undersample/years
    
    def _get_colour(self):
        """Returns the colour of the species."""
        if self.species == 'Black Hole':
            return 'tab:orange'
        elif self.species == 'Neutron Star':
            return 'tab:green'
        elif self.species == 'Star':
            return 'tab:blue'
        elif self.species == 'White Dwarf':
            return 'tab:red'
        else:
            # Pick a consistent random dash-dash line
            np.random.seed(hash(self.species) % 2**32)
            linestyle = (0, tuple(np.random.randint(1, 10, 4)))
            np.random.seed()
            return linestyle
    
    def _get_linestyle(self):
        """Returns the colour of the species."""
        if self.species == 'Black Hole':
            return 'dashed'
        elif self.species == 'Neutron Star':
            return 'dashdot'
        elif self.species == 'Star':
            return 'dotted'
        elif self.species == 'White Dwarf':
            # This is a dash-dot-dot line
            return (0, (3, 3, 1, 3, 1, 3))
        else:
            # Pick a consistent random RGB colour
            np.random.seed(hash(self.species) % 2**32)
            colour = tuple(np.random.uniform(0, 1, 3))
            np.random.seed()
            return colour
    
    def set_bright(self, bright_threshold):
        """Sets the bright attribute to be the events which are brighter than the bright_threshold."""
        self.bright_threshold = bright_threshold
        
        # In case the threshold is 0 or None, we want to keep all events (even events with null magntiudes)
        if bright_threshold is None or bright_threshold == 0:
            self.bright = self.events.copy()
        
        self.bright = self.events[self.events['bump magnitude'] > bright_threshold].copy()
        
def get_logger(logging_file=None, append_logging=False):
    """
    Get a logger which logs to a file or stdout if no file is provided.

    Parameters
    ----------
    logging_file : str, optional
        The filepath to save the logging to. If None, logging is printed to stdout.
    append_logging : bool, optional
        Whether to append to the logging file. Default is False.
    Returns
    -------
    logging.Logger
        The logger.
    """
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if logging_file is None:
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        if append_logging:
            handler = logging.FileHandler(logging_file, mode='a')
        else:
            handler = logging.FileHandler(logging_file, mode='w')
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(process)d - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
