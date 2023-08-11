
import pandas as pd
import numpy as np
from scipy import stats


class FitAlphaConstant:
    def __init__(self, file):
        self.file = file
        self.dataframe = self._get_data()
        self.vel_col, self.alpha_col = self._get_titles()
        self.describe_dataframe = self._filter_data_describe()

    def _get_data(self):
        """
        Record file and put into dataframe
        :param file: Excel file
        :return: dataframe
        """
        dataframe = pd.read_excel(self.file)
        return dataframe

    def _get_titles(self):
        """
        Get the column titles for use in pandas functions.
        :param diffdata: input dataframe
        :return: get the names for columns with recorded drift velocity,
        and the alpha value for the distribution of velocities
        """
        vel_col = self.dataframe.columns[3]
        alpha_col = self.dataframe.columns.values[1]
        return vel_col, alpha_col

    def _filter_data_describe(self):
        """
        This removes all the outliers which are 3*standard_deviation
        away from the mean, and then return a describe_dataframe.
        :param dataframe: Input dataframe.
        :param vel_col: Velocity data point column title.
        :param alpha_col: Alpha constant column title.
        :return: After the inputted dataframe is filtered, group by all
        the alpha constants to sort the velocity data and apply .describe().
        """
        # A list of filtered velocity values to be used.
        filtered_vel = pd.Series(dtype='float64')
        # Loop through all the alpha constants.
        for x in self.dataframe.alpha.unique():
            # Get the velocity values for current alpha constant.
            alpha_vel_val = self.dataframe[self.vel_col].loc[self.dataframe[self.alpha_col] == x]
            # Filter out all the values which are greater than 3*standard_deviation
            alpha_vel_val = alpha_vel_val.loc[stats.zscore(np.abs(alpha_vel_val)) <= 3]
            # Add accepted velocity values to the filtered_vel list.
            filtered_vel = pd.concat([filtered_vel, alpha_vel_val])
        # Update dataframe to only have velocity values that are in "filtered_vel".
        self.dataframe = self.dataframe[self.dataframe[self.vel_col].isin(filtered_vel)]
        return self.dataframe.groupby(self.alpha_col)[self.vel_col].describe()

    def find_optimal_constant(self, lowest_tstat=1000, pvalue=0.05):
        """
        Function to take in the describe_dataframe generated from the function,
        'filter_data_describe', and then loop through each alpha constant in its index.
        The alpha is used to filter all velocity values in vel_col column,
        which have the matching constant in the alpha row,
        to apply a one sample t-test on the velocity value data.

        NOTE: In this case 'alpha' is the equation constant,
        and NOT the threshold to test the p-value.
        :return: alpha, lowest_tstat, pvalue
        """
        alpha = self.dataframe['alpha'].mean()
        for x in self.describe_dataframe.index.values:
            tstat, p = stats.ttest_1samp(self.dataframe[self.dataframe['alpha'] == x][self.vel_col],
                                         popmean=self.dataframe.Theory_Vel.mean()) # Center around theoretical velocity.
            if np.abs(tstat) < abs(lowest_tstat):
                alpha = x
                lowest_tstat = tstat
                pvalue = p
        return alpha, lowest_tstat, pvalue


i = 30
file = f'pos_{i}_alpha_values_MaxC_60000_Grad_0.000405.xlsx'
find_alpha = FitAlphaConstant(file)
a, _, _ = find_alpha.find_optimal_constant()
print(a)
