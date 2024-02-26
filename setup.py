import setuptools 

REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'matplotlib',
    'ebf',
    'galpy',
    'astropy',
    'requests',
    'astroquery',
    'ray'
]

DESCRIPTION = "A package to synthesise populations of dead stars and calculate microlensing events caused by this population"

setuptools.setup( 
	name="stellarmortis",
	version="0.0.1",
	author="David Sweeney", 
	author_email="david.sweeney@sydney.edu.au", 
	packages=["stellarmortis"], 
	description=DESCRIPTION, 
	long_description=DESCRIPTION, 
	url="https://github.com/David-Sweeney/StellarMortis",
	license='MIT', 
	python_requires='>=3.8', 
	install_requires=REQUIRED_PACKAGES 
)
