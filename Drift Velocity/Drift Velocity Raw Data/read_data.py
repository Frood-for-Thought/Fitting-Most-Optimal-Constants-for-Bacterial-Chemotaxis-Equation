
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
    vel_data = Diffusion_Data.columns[3]
    alpha_const = Diffusion_Data.columns.values[1]
    return vel_data, alpha_const


def filter_data_describe(dataframe, vel_data, alpha_const):
    """
    This removes all the outliers which are 3*standard_deviation
    away from the mean, and then return a describe_dataframe.
    :param dataframe: Input dataframe.
    :param vel_data: Velocity data point column title.
    :param alpha_const: Alpha constant column title.
    :return:After the inputted dataframe is filtered, group by all
    the alpha constants to sort the velocity data and apply .describe().
    """
    # Filter out datapoints that are greater than 3xSD
    dataframe = dataframe[np.abs(stats.zscore(dataframe[vel_data]) <= 3)]
    return dataframe.groupby(alpha_const)[vel_data].describe()


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
                                     popmean=dataframe.Theory_Vel[0])  # Center around theoretical velocity.
        if np.abs(tstat) < abs(lowest_tstat):
            alpha = x
            lowest_tstat = tstat
            pvalue = p
    return alpha, lowest_tstat, pvalue


file = 'pos_30_alpha_values_MaxC_60000_Grad_0.000405.xlsx'
Diffusion_Data = pd.DataFrame(get_data(file))
pd.set_option('display.max_columns', None)  # Show all the columns
vel_data, alpha_const = get_titles(Diffusion_Data)
describe_diff_by_const = filter_data_describe(Diffusion_Data, vel_data, alpha_const)
print(describe_diff_by_const)
print(Diffusion_Data.Theory_Vel[0])
print(find_optimal_constant(describe_diff_by_const, Diffusion_Data, vel_data))


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



