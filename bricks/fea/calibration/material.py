# ---------------------------------------------------------------------------- #
#                                   Database                                   #
# ---------------------------------------------------------------------------- #

class MaterialParameters:
    def __init__(self):
        # Initialize material parameters with their mean and standard deviation
        self.parameters = {
            'fmy': (17.7, 6.73),
            'Emy': (9348, 3271),
            'fmx': (11.0, 2.53),
            'Emx': (5470, 547),
            'fw': (0.43, 0.07),
            'fx2': (1.22, 0.11),
            'fx3': (0.76, 0.21),
            'G': (1252, 550),
            'fv': (0.35, 0),
            'u': (0.67, 0)
        }

    def sample_parameter(self, param_name, percentile):
        """
        Samples the xth percentile value from a normal distribution for a given parameter.
        
        Args:
            param_name (str): The name of the parameter to sample (e.g., 'fw').
            percentile (float): The percentile to sample (0-100).
        
        Returns:
            float: The value at the given percentile in the distribution.
        """
        confidence = percentile / 100
        mean, std_dev = self.parameters[param_name]
        return norm.ppf(confidence, loc=mean, scale=std_dev)

    def get_bounds(self, param_name, percentile):
        """
        Generates the lower and upper bounds for a given parameter based on a confidence interval.
        
        Args:
            param_name (str): The name of the parameter (e.g., 'fw').
            confidence (float): The confidence level for the bounds (e.g., 0.95).
        
        Returns:
            tuple: A tuple containing the lower and upper bounds for the parameter.
        """
        confidence = percentile / 100
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        lower_bound = self.sample_parameter(param_name, lower_percentile)
        upper_bound = self.sample_parameter(param_name, upper_percentile)
        return lower_bound, upper_bound

    @staticmethod
    def tensile_strength(fw):
        """
        Calculates the tensile strength based on the provided fw value.
        
        Args:
            fw (float): The value of fw.
        
        Returns:
            float: The tensile strength.
        """
        return 0.8 * fw

    @staticmethod
    def tensile_fracture_energy(fw, mode):
        """
        Calculates the tensile fracture energy based on the provided fw value and mode.
        
        Args:
            fw (float): The value of fw.
            mode (str): The mode, either 'mortar' or 'brick'.
        
        Returns:
            float: The tensile fracture energy.
        """
        if mode == 'mortar':
            fm_mortar = fw / 0.036  # Mpa
            GfI = 0.025 * (fm_mortar / 10) ** 0.7  # N/mm
        elif mode == 'brick':
            GfI = 10 * 0.16 * MaterialParameters.tensile_strength(fw)  # N/mm
        return GfI

    @staticmethod
    def compressive_fracture_energy(fmy):
        """
        Calculates the compressive fracture energy based on the provided fmy value.
        
        Args:
            fmy (float): The value of fmy.
        
        Returns:
            float: The compressive fracture energy.
        """
        GfC = 3.09 * fmy
        return GfC

    @staticmethod
    def shear_fracture_energy(fm):
        """
        Calculates the shear fracture energy based on the provided fm value.
        
        Args:
            fm (float): The value of fm.
        
        Returns:
            float: The shear fracture energy.
        """
        Gft_m = 0.025 * (fm / 10) ** 0.7  # N/mm
        GfII = 10 * Gft_m  # N/mm
        return GfII
