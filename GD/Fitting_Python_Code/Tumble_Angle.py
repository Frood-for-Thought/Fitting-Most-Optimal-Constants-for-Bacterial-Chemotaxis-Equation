import numpy as np
from scipy.stats import rv_continuous


class Tumble_Angle_Distribution(rv_continuous):
    def _pdf(self, x):
        """
        Override the _pdf describe bacterial tumble angle.
        :param x: Angle in Rad.
        :return: P(x) = 0.5*(1+cos(x))*sin(x)
        """
        return 0.5 * (1 + np.cos(x)) * np.sin(x)


class New_Angle(Tumble_Angle_Distribution):
    def __init__(self):
        # Initialize the base class with angles from 0 to pi radians.
        super().__init__(a=0, b=np.pi, name='custom_angle_distribution')

    def tumble_angle_function(self):
        """
        Randomly select a new tumble angle from the probability distribution, P(x) = 0.5*(1+cos(x))*sin(x).
        :return: The repositioned tumble angle in degrees.
        """
        random_angle_rad = self.rvs()
        random_angle_deg = np.degrees(random_angle_rad)
        return random_angle_deg
