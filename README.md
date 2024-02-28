# StellarMortis

[![PyPI version](https://badge.fury.io/py/stellarmortis.svg)](https://badge.fury.io/py/stellarmortis)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 

#### Note that this package is still under development and, although functional, is not yet fully documented. If you have any questions or need help using the package, please feel free to contact me (my email is given in the paper below). I will be happy to help you get started.
 
StellarMortis, named after *rigor mortis*, is a Python package which allows you to create populations of dead stars&mdash;*galactic underworlds*&mdash;and to simulate their effects as seen through microlensing.

This package combines the code developed in *[The Galactic Underworld](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4971S/abstract)* and *Observing the Galactic Underworld* (currently under review) with significant modifications and extensions. Particularly, the code has been restructured to be more modular and to allow the population of dead stars generated to have their microlensing events seemlessly simulated.

## Installation
To install StellarMortis simply use pip:

    pip install stellarmortis

## Quick Start
StellarMortis is designed to be as simple to use as possible. The following example demonstrates how to create a population of black holes and simulate their microlensing events.

    import stellarmortis as sm

    masses = sm.Mass()
    natal_kicks = sm.NatalKick()
    guw = sm.Underworld('path/to/guw_f1e-4_bhm2.35.ebf', natal_kicks=natal_kicks, 
                        masses=masses, species=['Black Hole', 'Neutron Star'], 
                        logging_file='test.log', verbose=1)

    guw.evolve()
    guw.save('NSs_and_BHs.csv')

    guw.calculate_microlensing(run_name='example', output_filepath='./GUW_events.ecsv', 
                               years=100, num_workers=-1, collate=True)
    guw.plot_microlensing(undersamples={'Black Hole': 1e4,
                                        'Neutron Star': 1e4}, output_dir='./plots')

For a more detailed explanation of the code, see the sections below.

## Underworld Synthesis
StellarMortis assumes you have some prexisting population of dead stars that you want to apply a natal kick to and evolve through the Galactic potential. Typically, this population is generated using a stellar evolution code such as [GALAXIA](https://ui.adsabs.harvard.edu/abs/2011ApJ...730....3S/abstract). The publicly available version of GALAXIA only outputs the populations of visible galaxies, so we have made available two populations generated using a modified version of GALAXIA which outputs only dead stars. These populations are available for download in the `data` folder of this repository. We also include two populations of visible stars which were created using the publicly available version of GALAXIA, also available in the `data` folder. They are:

- `guw_f1e-4_bhm2.35.ebf` &mdash; A population of dead stars which is undersampled by $10^4$.
- `guw_f1e-3_bhm2.35.ebf` &mdash; A population of dead stars which is undersampled by $10^3$.
- `milkyway_f1e-6.ebf` &mdash; A population of visible which is undersampled by $10^6$.
- `milkyway_f1e-4.ebf` &mdash; A population of visible which is undersampled by $10^4$.

To use these files, simply download them and instantiate an `Underworld` object, providing a filepath to the downloaded file:

    import stellarmortis as sm
    guw = sm.Underworld('path/to/guw_f1e-4_bhm2.35.ebf')

By default the population is tagged as black holes, neutron stars and white dwarfs based on the initial mass, $m_p$, of their progenitor star (provided by GALAXIA). If $m_p < 8 M_{\odot}$, the star is tagged as a white dwarf. If $8 M_{\odot} \leq m_p < 25 M_{\odot}$, the star is tagged as a neutron star. If $m_p \geq 25 M_{\odot}$, the star is tagged as a black hole. This can be changed by accessing the `data` attribute of the `Underworld`, which is a `pandas.DataFrame` object containing the columns `smass` (the initial mass of the progenitor star $m_p$), `feh` (the metallicity of the progenitor star) `species` (the species of the dead star) and `px`, `py`, `pz`, `vx`, `vy`, `vz` (the position and velocity of the dead star).

This `data` attribute can be modified to make whatever changes are desired (or just for inspection). For example, tagging all remnants with $m_p \geq 20 M_{\odot}$ as black holes can be done with the following line of code:

    guw.data.loc[guw.data['smass'] >= 20, 'species'] = 'Black Hole'

To filter the population to only include certain species of remnant the `.filter_species()` method may be used:

    guw.filter_species(['Black Hole', 'Neutron Star'])

This will remove all objects which are not tagged as a `Black Hole` or `Neutron Star` from the population.

### Masses

To specify the masses of the remnants the `Mass` class is used. This class is instantiated with a dictionary containing the masses of the remnants. The keys of the dictionary are the species of dead star and the values are the mass of the remnant. The masses can be specified as a single value in solar masses or a function which takes the initial mass of the progenitor star ($m_p$) as an argument and returns the mass of the remnant in solar masses. 

To add these masses to the population, the `Mass` object is passed to the `.add_masses()` method of the `Underworld` object.

For example, to specify the masses of the remnants as $1.35 M_{\odot}$ for neutron stars and $7.8 M_{\odot}$ for black holes, the following code can be used:

    masses = sm.Mass({'Neutron Star': 1.35, 'Black Hole': 7.8})
    guw.add_masses(masses)

By default a `Mass` object will be instantiated with the masses as described above. If the masses are specified when the `Underworld` object is instantiated then the masses are added immediately (see the [Quick Start](#quick-start) example).

In general, if you do not intend to calculate the microlensing events caused by the population then you will not need to specify the mass of the remnants.

### Natal Kicks
As demonstrated in *[The Galactic Underworld](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4971S/abstract)*, natal kicks play a big role in the distribution of the galactic underworld. The `NatalKick` class provides a simple interface to apply natal kicks to the population of dead stars. The `NatalKick` class is instantiated with a dictionary containing the natal kicks to be applied to each species of dead star. The keys of the dictionary are the species of dead star and the values are the natal kicks to be applied. The natal kick can be specified as a single value in km/s, a string specifiying one of the pre-defined natal kicks or a function which takes the mass of the remnant ($m_r$) and the initial mass of the progenitor star ($m_p$) as arguments and returns the natal kick in km/s. The pre-defined natal kicks are:

- `Hobbs2005` &mdash; The natal kicks from [Hobbs et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005MNRAS.360..974H/abstract).
- `Igoshev2020` &mdash; The natal kicks from [Igoshev et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.3663I/abstract) for the young pulsars.
- `Igoshev2020All` &mdash; The natal kicks from [Igoshev et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.3663I/abstract) for all measured pulsars.
- `Renzo2018` &mdash; The natal kicks from [Renzo et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019A&A...624A..66R) which describe the natal kicks experienced by entire neutron star binary systems in which the neutron star remains in the binary.

Each pre-defined natal kick distribution also has a corresponding `scaled` version which scales the natal kicks by a factor of $1.35/m_r$. This is intended to be used on the black holes if the natal kicks are to be scaled by the mass of the remnant (i.e. to provide the same momentum to all neutron stars and black holes). $1.35 M_{\odot}$ is taken to be the average mass of a neutron star in solar masses.

These natal kicks can then be applied to the population using the `.add_kicks()`. 

For example, to apply the natal kicks from [Igoshev et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.1420I/abstract) to a population of black holes and neutron stars, the following code can be used:

    natal_kicks = sm.NatalKick({'Neutron Star': 'Igoshev2020', 
                                'Black Hole': 'scaled Igoshev2020'})
    guw.add_kicks(natal_kicks)

By default a `NatalKick` object will be instantiated with the above natal kick distributions. If the natal kicks are specified when the `Underworld` object is instantiated then the kicks are applied immediately (see the [Quick Start](#quick-start) example).

### Logging
Logging messages will be printed to the console by default. To change the quantity of information displayed during execution the `verbose` parameter can be set to `0`, `1`, `2` or `3`. Setting `verbose` to `0` will suppress all logging messages. Setting `verbose` to `1` will display the default logging messages. Setting `verbose` to `2` or `3` will display all futher logging messages, intended for debugging. Verbose can be set when instantiating the `Underworld` object or by setting the `verbose` attribute at any time.

The logging messages can also be saved to a file by setting the `logging_file` parameter to a filepath. This will save all logging messages to the specified file. If a logging file is to be set it must be done when the `Underworld` object in instantiated.


### Evolution

Once the desired birth population is reached, it can be evolved through the `MWPotential2014` Galactic potential using [galpy](https://ui.adsabs.harvard.edu/abs/2015ApJS..216...29B/abstract) using the `.evolve()` method:

    guw.evolve()

This will evolve the population through the `MWPotential2014` Galactic potential for a duration unique to each remnant. The duration is the estimated time since each remnant was "born" upon the death of its progenitor star.

### Saving

The population can be saved to a file using the `.save()` method. This will save the population to a CSV file. The filepath of the file to be saved to must be specified. For example:

    guw.save('GUW_events.csv')

The file must be saved as a CSV file prior to calculating the microlensing events caused by the population.

## Microlensing
StellarMortis also provides a simple interface to simulate the microlensing of a population of dead stars. This is done by performing an N-body simulation of the population and a population of lenses. 

### Continuing from an `Underworld` object
If the microlensing events are to be calculated immediately after the population is evolved then the `Underworld` object can be used to calculate the microlensing events. This is done by calling the `.calculate_microlensing()` method. The method takes the following parameters:

- `run_name` &mdash; The name of the run. This will be used to name the output file.
- `years` &mdash; The duration of the microlensing simulation in years.
- `collate` &mdash; If `True` then the output file will be collated into a single file. If `False` then the output file will be saved as a separate file for each species of dead star.
- `output_filepath` &mdash; The filepath of the output file. The output file will be saved as an ECSV file.
- `progress_dir` &mdash; The directory to save the progress files to. If `None` then no progress files will be saved.
- `sensitivity` &mdash; The sensitivity of the microlensing event. This is the Einstein radius in microarcseconds.
- `num_workers` &mdash; The number of workers to use. If set to `-1` then the number of workers will be set to the number of cores on the machine.
- `start` &mdash; The index of the first microlensing event to be simulated.
- `end` &mdash; The index of the last microlensing event to be simulated.

For example:

    guw.calculate_microlensing(run_name='example', output_filepath='./BH_events.ecsv', 
                               years=100, num_workers=-1, collate=True)

This will calculate the microlensing events caused by the population of black holes and neutron stars and save the output to `./BH_events.ecsv`.

### Starting from a CSV file
If the microlensing events are to be calculated at a later time then the microlensing events can be calculated from a CSV file. This is done by instantiating a `Microlensing` object and calling the `calculate_microlensing()` function. The function takes the following parameters:

- `filepath` &mdash; The filepath of the CSV file containing the microlensing events.
- `run_name` &mdash; The name of the run. This will be used to name the output file.
- `years` &mdash; The duration of the microlensing simulation in years.
- `collate` &mdash; If `True` then the output file will be collated into a single file. If `False` then the output file will be saved as a separate file for each species of dead star.
- `output_filepath` &mdash; The filepath of the output file. The output file will be saved as an ECSV file.
- `progress_dir` &mdash; The directory to save the progress files to. If `None` then no progress files will be saved.
- `sensitivity` &mdash; The sensitivity of the microlensing event. This is the Einstein radius in microarcseconds.
- `num_workers` &mdash; The number of workers to use. If set to `-1` then the number of workers will be set to the number of cores on the machine.
- `start` &mdash; The index of the first microlensing event to be simulated.
- `end` &mdash; The index of the last microlensing event to be simulated.
- `logger` &mdash; The logger to use. If `None` then a new logger will be created.
- `logging_file` &mdash; The filepath of the logging file. If `None` then no logging file will be created.
- `append_logging` &mdash; If `True` then the logging file will be appended to. If `False` then the logging file will be overwritten.
- `verbose` &mdash; The verbosity of the logging messages.

For example:

    ml = sm.Microlensing('GUW_events.csv')
    ml.calculate_microlensing(run_name='example', output_filepath='./BH_events.ecsv', 
                              years=100, num_workers=-1, collate=True)

This will calculate the microlensing events caused by the population of black holes and neutron stars and save the output to `./BH_events.ecsv`.

## Plotting

### Continuing from an `Underworld` object

The microlensing events can be plotted using the `.plot_microlensing()` method. The method takes the following parameters:

- `undersamples` &mdash; A dictionary containing the undersampling factor for each species of dead star. The keys of the dictionary are the species of dead star and the values are the undersampling factor. The undersampling factor is the number of microlensing events to be plotted. For example, if the undersampling factor is $10^4$ then only $1$ in $10^4$ microlensing events will be plotted.
- `other_filepaths` &mdash; A list of filepaths to other microlensing event files to be plotted.
- `output_dir` &mdash; The directory to save the plots to.
- `bootstraps` &mdash; The number of bootstraps to perform. If set to `0` then no bootstraps will be performed.
- `trim_data` &mdash; If `True` then the data will be trimmed to the same length. If `False` then the data will not be trimmed.
- `save_summary` &mdash; If `True` then the summary statistics will be saved to a file.

For example:

    guw.plot_microlensing(undersamples={'Black Hole': 1e4,
                                        'Neutron Star': 1e4}, output_dir='./plots')

This will plot the microlensing events caused by the population of black holes and neutron stars and save the plots to `./plots`.

### Starting from a CSV/ECSV file

The microlensing events can be plotted using the `plot_microlensing()` function. This function creates a range of summary plots of the microlensing events.

The function takes the following parameters:

- `filepaths` &mdash; A list of filepaths to the microlensing event files to be plotted.
- `undersamples` &mdash; A dictionary containing the undersampling factor for each species of dead star. The keys of the dictionary are the species of dead star and the values are the undersampling factor. The undersampling factor is the number of microlensing events to be plotted. For example, if the undersampling factor is $10^4$ then only $1$ in $10^4$ microlensing events will be plotted.
- `output_dir` &mdash; The directory to save the plots to.
- `bootstraps` &mdash; The number of bootstraps to perform. If set to `0` then no bootstraps will be performed.
- `trim_data` &mdash; If `True` then the data will be trimmed to the same length. If `False` then the data will not be trimmed.
- `save_summary` &mdash; If `True` then the summary statistics will be saved to a file.
- `logger` &mdash; The logger to use. If `None` then a new logger will be created.
- `logging_file` &mdash; The filepath of the logging file. If `None` then no logging file will be created.
- `append_logging` &mdash; If `True` then the logging file will be appended to. If `False` then the logging file will be overwritten.
- `verbose` &mdash; The verbosity of the logging messages.

## Citation

My intention is to publish this package in a peer-reviewed journal. In the meantime, if you use this package in your research, please let me know &mdash; I will upload the package to Zenodo so that you can cite it directly. Please also cite the following papers which describe the details of the simulations performed by the package:

- [The Galactic Underworld](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4971S/abstract)
- *Observing the Galactic Underworld* (currently under review)

Please let me know if you use this package. I would love to hear about it and get any feedback you might have!



