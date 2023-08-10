
import pandas as pd
import numpy as np
from scipy import stats


def get_data(file):
    """
    Record file and put into dataframe
    :param file: Excel file
    :return: dataframe
    """
    dataframe = pd.read_excel(file)
    return dataframe


def get_titles(dataframe):
    '''
    Get the column titles for use in pandas functions.
    :param diffdata: input dataframe
    :return: get the names for columns with recorded drift velocity,
    and the alpha value for the distribution of velocities
    '''
    vel_data = dataframe.columns[3]
    alpha_const = dataframe.columns.values[1]
    return vel_data, alpha_const


def filter_data_describe(dataframe, vel_col, alpha_col):
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
    for x in dataframe.alpha.unique():
        # Get the velocity values for current alpha constant.
        alpha_vel_val = dataframe[vel_col].loc[dataframe[alpha_col] == x]
        # Filter out all the values which are greater than 3*standard_deviation
        alpha_vel_val = alpha_vel_val.loc[stats.zscore(np.abs(alpha_vel_val)) <= 3]
        # Add accepted velocity values to the filtered_vel list.
        filtered_vel = pd.concat([filtered_vel, alpha_vel_val])
    # Update dataframe to only have velocity values that are in "filtered_vel".
    dataframe = dataframe[dataframe[vel_col].isin(filtered_vel)]
    return dataframe.groupby(alpha_col)[vel_col].describe(), dataframe


def find_optimal_constant(describe_dataframe, dataframe, vel_data, lowest_tstat=1000, pvalue=0.05):
    """
    Function to take in the describe_dataframe generated from the function,
    'filter_data_describe', and then loop through each alpha constant in its index.
    The alpha is used to filter all velocity values in vel_data column,
    which have the matching constant in the alpha row,
    to apply a one sample t-test on the velocity value data.

    NOTE: In this case 'alpha' is the equation constant,
    and NOT the threshold to test the p-value.
    :return: alpha, lowest_tstat, pvalue
    """
    alpha = dataframe.loc[0, 'alpha']
    for x in describe_dataframe.index.values:
        tstat, p = stats.ttest_1samp(dataframe[dataframe['alpha'] == x][vel_data],
                                     popmean=dataframe.Theory_Vel.mean())  # Center around theoretical velocity.
        if np.abs(tstat) < abs(lowest_tstat):
            alpha = x
            lowest_tstat = tstat
            pvalue = p
    return alpha, lowest_tstat, pvalue


file = 'pos_30_alpha_values_MaxC_60000_Grad_0.000405.xlsx'
Alpha_Vel_Data = pd.DataFrame(get_data(file))
pd.set_option('display.max_columns', None)  # Show all the columns
vel_col, alpha_col = get_titles(Alpha_Vel_Data)
describe_diff_by_const, Alpha_Vel_Data = filter_data_describe(Alpha_Vel_Data, vel_col, alpha_col)
print(describe_diff_by_const)
print(Alpha_Vel_Data.Theory_Vel.mean())
print(find_optimal_constant(describe_diff_by_const, Alpha_Vel_Data, vel_col))


print("If the p value is large then the null hypothesis cannot be rejected and there is no evidence of a difference")

# lowest_tstat = 10000000
# alpha = Diffusion_Data.loc[0, 'alpha']
# pvalue = 0.05
# for x in describe_diff_by_const.index.values:
#     print(f"alpha: {x}")
#     print(f"Calculated Mean: {Diffusion_Data[Diffusion_Data['alpha'] == x]['Calc_Velocity'].mean()}")
#     print(f"Theoretical Mean: {Diffusion_Data.loc[0, 'Theory_Vel']}")
#     tstat, p = stats.ttest_1samp(Diffusion_Data[Diffusion_Data['alpha'] == x]['Calc_Velocity'], popmean=Diffusion_Data.Theory_Vel[0])
#     print(f"T-stat results: {np.abs(tstat), p}\n")
#     if np.abs(tstat) < abs(lowest_tstat):
#         alpha = x
#         lowest_tstat = tstat
#         pvalue = p
# print(alpha, lowest_tstat, pvalue)



