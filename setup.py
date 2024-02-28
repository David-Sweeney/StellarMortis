import setuptools 

REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'ebfpy',
    'galpy',
    'astropy',
    'requests',
    'astroquery',
    'ray'
]

DESCRIPTION = "A package to synthesise populations of dead stars and calculate microlensing events caused by this population."
LONG_DESCRIPTION = \
    """A package to synthesise populations of dead stars and calculate microlensing events caused by this population.
    
    For more information, please visit the GitHub page: https://github.com/David-Sweeney/StellarMortis"""

setuptools.setup( 
	name="stellarmortis",
	version="0.0.8",
	author="David Sweeney", 
	author_email="david.sweeney@sydney.edu.au", 
	packages=setuptools.find_packages(), 
	description=DESCRIPTION, 
	long_description=DESCRIPTION, 
	url="https://github.com/David-Sweeney/StellarMortis",
	license='MIT', 
	python_requires='>=3.7', 
	install_requires=REQUIRED_PACKAGES,
)
