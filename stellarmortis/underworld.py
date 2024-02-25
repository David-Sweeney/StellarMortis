import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import astropy.units as u

from . import galaxy
from . import microlensing
from . import kick
from . import mass
from .utils import get_logger, MicrolensingTable

class Underworld():
    def __init__(self, filepath, natal_kicks=None, masses=None, species=None, logging_file=None, append_logging=False, verbose=1):
        """Initialise the DarkGalaxy class
        
        If natal kicks are provided, they will be applied to the galaxy. If masses are 
        provided, they will be applied to the galaxy. If species are provided, only 
        those species will be included in the galaxy.
        
        Args
        ----
        filepath : str
            The filepath to the galaxy data.
        natal_kicks : NatalKick or float or int or callable, optional
            The natal kicks to apply to the galaxy. If None, no natal kicks are applied.
        masses : Mass or float or int or callable, optional
            The masses to apply to the galaxy. If None, no masses are applied.
        species : str or list of str, optional
            The species to include in the galaxy. If None, all species are included.
            
        Properties
        ----------
        filepath : str
            The filepath to the galaxy data.
        natal_kicks : NatalKick or None
            The natal kicks to apply to the galaxy. If None, no natal kicks have been applied.
        masses : Mass or None
            The masses to apply to the galaxy. If None, no masses have been applied.
        data : pandas.DataFrame
            The galaxy data.
        saved : bool
            Whether the galaxy data has been saved.
        """
        self.filepath = filepath
        self.natal_kicks = natal_kicks
        self.masses = masses
        self.species = species
        self.verbose = verbose
        self.microlensing_filepath = None
        self.logger = get_logger(logging_file, append_logging)
        
        self.data, self.saved = self.load_data(self.filepath)
        
        if self.species is not None:
            self.filter_species(self.species)
        if self.masses is not None:
            self.add_masses(self.masses)
        if self.natal_kicks is not None:
            self.add_kicks(self.natal_kicks)

    def load_data(self, filepath):
        """
        Load the galaxy data from a file.

        Parameters
        ----------
        filepath : str
            The filepath to the galaxy data.
        """
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
            saved = True
        elif filepath.endswith('.ebf'):
            data = galaxy.load_data(filepath)
            saved = False
        else:
            raise ValueError(f'Filepath: {filepath} must end with .csv or .ebf')
            
        if self.verbose >= 1:
            self.logger.info(f'Loaded galaxy data from {filepath}')
        
        return data, saved
                   
    def filter_species(self, species):
        """Filter the galaxy data by species.
        
        Args:
        -----
        species : str or list of str
            The species to include in the galaxy.
        """
        
        if isinstance(species, str):
            species = [species]
            
        old_species = pd.unique(self.data['rtype'])
            
        keep = np.array([False]*len(self.data))
        for s in species:
            keep = keep | (self.data['rtype'] == s)
            
        self.data = self.data[keep]
        self.saved = False
        self.species = species
        
        if self.verbose >= 1:
            self.logger.info(f'Underworld data previously contained {old_species} ' 
                             f'but has been filtered to contain species: {species}')
    
    def add_masses(self, masses):
        """Add masses to the galaxy data.
        
        Args:
        -----
        masses : Mass or float or int or callable
            The masses to apply to the galaxy.
        """
        
        if not isinstance(masses, mass.Mass):
            masses = mass.Mass(masses)
        self.data = galaxy.add_masses(self.data, masses)
        self.saved = False
        self.masses = masses
        
        if self.verbose >= 1:
            self.logger.info(f'Added masses to galaxy data: {masses}')
    
    def add_kicks(self, natal_kicks):
        """Add natal kicks to the galaxy data.

        Args:
        -----
        natal_kicks : NatalKick or float or int or callable
            The natal kicks to apply to the galaxy.
        verbose : int, optional
            The verbosity level. Default is 0.
        """
        
        if not isinstance(natal_kicks, kick.NatalKick):
            natal_kicks = kick.NatalKick(natal_kicks)
            
        data = galaxy.add_kicks(self.data, natal_kicks, logger=self.logger, verbose=self.verbose)
        data = galaxy.update_cylindrical_coords(data)
        self.data = galaxy.tag_escaping(data)
        self.saved = False
        self.natal_kicks = natal_kicks
        
        if self.verbose >= 1:
            self.logger.info(f'Added natal kicks to galaxy data: {natal_kicks}')

    def evolve(self, sections=None):
        """Evolve the galaxy data.

        Args:
        -----
        sections : int, optional
            The number of sections to split the data into. Default is None.
        verbose : int, optional
            The verbosity level. Default is 0.
        """
        df = self.data
        
        # Data must be split into sections to cope with memory restrictions
        if sections is None:
            sections = int(len(self.data)//5e4 + 1)
        
        section_length = len(df)//sections + 1
        section_dfs = [df.iloc[i*section_length:(i+1)*section_length].copy() for i in range(sections)]
        for i in range(sections):
            if self.verbose >= 1:
                self.logger.info('*'*20)
                self.logger.info(f'Dataframe {i+1}/{sections}')
            section_dfs[i].loc[:, 'velocity'] = np.sqrt(np.sum(section_dfs[i].loc[:, ['vx', 'vy', 'vz']]**2, axis=1))
            section_dfs[i] = galaxy.calculate_orbits(section_dfs[i], logger=self.logger, verbose=self.verbose)
            section_dfs[i] = galaxy.tag_escaping(section_dfs[i])
            
        self.data = pd.concat(section_dfs)
        self.saved = False

    def save(self, filepath):
        """Save the galaxy data to a file.
        
        Args:
        -----
        filepath : str
            The filepath to save the galaxy data to.
        """
        assert filepath.endswith('.csv'), 'Filepath must end with .csv'
        self.data.to_csv(filepath, index=False)
        self.filepath = filepath
        self.saved = True
        
        if self.verbose >= 1:
            self.logger.info(f'Saved galaxy data to {filepath}')
    
    def calculate_microlensing(self, *args, **kwargs):
        """Calculate microlensing for the galaxy data.
        
        Args:
        -----
        run_name : str
            The name of the run.
        years : int or float or list of int or float
            The number of years to observe for.
        num_workers : int, optional
            The number of workers to use. Default is 1.
        collate : bool, optional
            Whether to collate the microlensing data. Default is True.
        output_filepath : str, optional
            The filepath to save the microlensing data to. Default is None.
            If None, filepath is set to the same directory as the filepath with the filename
            combined with the run name.
        progress_dir : str, optional
            The directory to save the progress files to. If None, the directory of the
            output_filepath is used.
        sensitivity : astropy Angle, optional
            The sensitivity of the simulation to microlensing events. Default is 1 microarcsecond.
        verbose : int, optional
            The verbosity level. Default is 0.
        """
        
        # Check to see if the file has actually been saved
        if not self.saved:
            raise ValueError('The galaxy data must be saved before microlensing calculations can be performed.')
        
        output_filepath = calculate_microlensing(self.filepath, *args, logger=self.logger, verbose=self.verbose, **kwargs)
        self.microlensing_filepath = output_filepath
        return output_filepath

    def plot_microlensing(self, undersamples, other_filepaths=None, **kwargs):
        """
        Make a wide range of plots for microlensing data.

        Parameters
        ----------
        undersamples : dict of {str: int}
            A dictionary of species and undersample rates formatted as {species: undersample}.
        other_filepaths : dict of {str: str}, optional
            Use this dictionary to include microlensing events from other microlensing simulations. 
            This dictionary will overwrite any of the filepaths from this run if they share a 
            common species. The dictionary as {species: filepath}. If None, the filepath saved from
            Underworld.calculate_microlensing() is used.
        output_dir : str, optional
            The directory to save the plots to. If None, the directory of the galaxy data is used.
            Default is None.
        bootstraps : int or float, optional
            Number of bootstraps to perform. Set bootstraps to None to reuse a previous 
            bootstrap file in output_dir.  If a float is provided it is converted to an int. 
            The default is 1000.
        trim_data : bool, optional
            Whether to trim the data of the first and final years which typically have higher rates 
            of microlensing events. Default is True.
        save_summary : bool, optional
            Whether to save a summary of the microlensing data. Default is True.
        """
        assert not (self.microlensing_filepath is None and other_filepaths is None), 'Microlensing data must be calculated before plotting'
        
        filepaths = {}
        
        if self.microlensing_filepath is not None:
            filepaths = {species: self.microlensing_filepath for species in self.species}
        
        if other_filepaths is not None:
            filepaths.update(other_filepaths)
        
        plot_microlensing(filepaths, undersamples, logger=self.logger, verbose=self.verbose, **kwargs)

def calculate_microlensing(filepath, run_name, years, collate=True, output_filepath=None, 
                           progress_dir=None, sensitivity=1*u.uas, num_workers=1, start=0, end=None,
                           logger=None, logging_file=None, append_logging=True, verbose=0):
    """Calculate microlensing for a given galaxy data file.
    
    Args:
    -----
    filepath : str
        The filepath to the galaxy data.
    run_name : str
        The name of the run.
    years : int or float or list of int or float
        The number of years to observe for.
    collate : bool, optional
        Whether to collate the microlensing data into one file. Default is True.
    output_filepath : str, optional
        The filepath to which to save the collated microlensing data. Default is None. 
        If None, filepath is set to the same directory as the filepath with the filename 
        combined with the run name.
    progress_dir : str, optional
        The directory to save the progress files to. If None, the directory of the
        output_filepath is used.
    sensitivity : astropy Angle, optional
        The sensitivity of the simulation to microlensing events. Default is 1 microarcsecond.
    num_workers : int, optional
        The number of workers to use. Default is 1.
    start : int, optional
        The index to start at. Default is 0.
    end : int, optional
        The index to end at. Default is None.
    logger : logging.Logger, optional
        The logger to use. If None, a new logger is created.
    logging_file : str, optional
        The filepath to save the logging to. If None and logger is None, logging is 
        printed to stdout.
    append_logging : bool, optional
        Whether to append to the logging file. Default is True.
    verbose : int, optional
        The verbosity level. Default is 0.
    """
    
    if output_filepath is None:
        output_filepath = filepath.split('.')[0] + '_' + run_name + '.ecsv'
    if progress_dir is None:
        progress_dir = os.path.dirname(output_filepath)
        if progress_dir == '':
            progress_dir = '.'
    assert os.path.isdir(progress_dir), f'Progress directory: "{progress_dir}" does not exist'
    if logger is None:
        logger = get_logger(logging_file, append_logging)
    num_workers = int(num_workers)
    if num_workers == -1:
        num_workers = cpu_count()
    assert num_workers > 0, f'Number of workers must be greater than 0, not: {num_workers}'
    
    # Should remove lenses > 20 kpc away *** TODO
    step_size = microlensing.parallelised_main(filepath, progress_dir, years, sensitivity, run_name, start=start, end=end, 
                                               num_workers=num_workers, logger=logger, verbose=verbose)
    
    if collate:
        if end is None:
            end = microlensing.get_file_length(filepath) - 1
        index_range = list(range(start, end, step_size))
        index_range.append(end)
        
        progress_filepaths = []
        for file_start, file_end in list(zip(index_range[:-1], index_range[1:])):
            progress_filepaths.append(f'{progress_dir}/{run_name}_{file_start}-{file_end}.ecsv')
        
        microlensing.collate_files(progress_filepaths, progress_dir, output_filepath, 
                                    run_name=run_name, delete_progress=True, delete_task_dir=True, logger=logger)
    return output_filepath

def plot_microlensing(filepaths, undersamples, output_dir=None, bootstraps=1000, trim_data=True, 
                      save_summary=True, logger=None, logging_file=None, append_logging=True, verbose=1):
    """
    Make a wide range of plots for microlensing data specified by a dictionary of filepaths.

    Parameters
    ----------
    filepaths : dict of {str: str}
        A dictionary of species and filepaths formatted as {species: filepath}.
    undersamples : dict of {str: int}
        A dictionary of species and undersample rates formatted as {species: undersample}.
    output_dir : str, optional
        The directory to save the plots to. If None, the directory of one of the filepaths is used.
        Default is None.
    bootstraps : int or float, optional
        Number of bootstraps to perform. Set bootstraps to None to reuse a previous 
        bootstrap file in output_dir.  If a float is provided it is converted to an int. 
        The default is 1000.
    trim_data : bool, optional
        Whether to trim the data of the first and final years which typically have higher rates 
        of microlensing events. Default is True.
    save_summary : bool, optional
        Whether to save a summary of the microlensing data. Default is True.
    """
    
    # Set output dir to the same as one of the filepaths if not specified
    if output_dir is None:
        output_dir = os.path.dirname(list(filepaths.values())[0])
        if output_dir == '':
            output_dir = '.'
    if logger is None:
        logger = get_logger(logging_file, append_logging)
    
    tables = []
    for species, filepath in filepaths.items():
        if not os.path.isfile(filepath):
            raise ValueError(f'Filepath: {filepath} does not exist')
        if not filepath.endswith('.ecsv'):
            raise ValueError(f'Filepath: {filepath} must end with .ecsv')
        
        tables.append(MicrolensingTable(species, filepath, undersamples[species], trim=trim_data))
    
    microlensing.plot_all(tables, output_dir=output_dir, bootstraps=bootstraps, save_summary=save_summary, logger=logger, verbose=verbose)
