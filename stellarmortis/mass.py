class Mass():
    def __init__(self, distributions={'Black Hole': 7.8, 'Neutron Star': 1.35}):
        """
        Class to handle mass distributions.
        
        Parameters
        ----------
        distributions : dict
            Dictionary of mass distributions. The key is the species and the value is the mass. 
            The value can be a float, int or a function. If the value is a float or int, the mass
            is assumed to be constant with this value. If the value is a function, the function 
            should return the mass. The function should take no arguments.       
        """
        
        self.string_representation = str(distributions)
        self.distributions = distributions
    
    def get_mass(self, species, progenitor_mass):
        if isinstance(self.distributions[species], (int, float)):
            return self.distributions[species]
        elif callable(self.distributions[species]):
            return float(self.distributions[species](progenitor_mass))
        else:
            raise ValueError('Mass distribution for {species} not understood.')
    
    def __str__(self):
        return self.string_representation
        