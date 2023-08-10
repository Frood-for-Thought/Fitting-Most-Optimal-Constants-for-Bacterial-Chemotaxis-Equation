
import pandas as pd
import numpy as np
from scipy import stats

file = 'pos_30_alpha_values_MaxC_60000_Grad_0.000405.xlsx'


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


def filter_data(dataframe, vel_dp, alpha):
    """

    :param dataframe:
    :param vel_dp:
    :param alpha:
    :return:
    """
    # Filter out datapoints that are greater than 3xSD
    dataframe = dataframe[np.abs(stats.zscore(dataframe[vel_data]) <= 3)]
    describe_diff_by_const = dataframe.groupby(alpha)[vel_dp].describe()




Diffusion_Data = pd.DataFrame(get_data(file))
pd.set_option('display.max_columns', None)  # Show all the columns
vel_data, alpha_const = get_titles(Diffusion_Data)

describe_diff_by_const = Diffusion_Data.groupby(alpha_const)[vel_data].describe()
# Filter out datapoints that are greater than 3xSD
Diffusion_Data = Diffusion_Data[np.abs(stats.zscore(Diffusion_Data[vel_data]) <= 3)]
describe_diff_by_const = Diffusion_Data.groupby(alpha_const)[vel_data].describe()
print(describe_diff_by_const)
print(Diffusion_Data.Theory_Vel[0])

# 95% of sample means will be expected to lie within a confidence interval of ±1.96 standard errors of the sample mean
describe_diff_by_const['std_error'] = describe_diff_by_const['std']/np.sqrt(describe_diff_by_const['count'])
print(describe_diff_by_const)
describe_diff_by_const['95%_conf_int'] = describe_diff_by_const['std_error'] * 1.96
# Filter out the means that are not within the 95% confidence interval of the standard error.
describe_diff_by_const = describe_diff_by_const[np.abs(describe_diff_by_const['mean']) < describe_diff_by_const['95%_conf_int']]
describe_diff_by_const_list = describe_diff_by_const.reset_index().iloc[:, 0]


print(describe_diff_by_const.sort_values(by=['mean', 'std'], ascending=True))

lowest_tstat = 10000000
alpha = Diffusion_Data.loc[0, 'alpha']
pvalue = 0.05
for x in describe_diff_by_const.index.values:
    print(f"alpha: {x}")
    print(f"Calculated Mean: {Diffusion_Data[Diffusion_Data['alpha'] == x]['Calc_Velocity'].mean()}")
    print(f"Theoretical Mean: {Diffusion_Data.loc[0, 'Theory_Vel']}")
    tstat, p = stats.ttest_1samp(Diffusion_Data[Diffusion_Data['alpha'] == x]['Calc_Velocity'], popmean=Diffusion_Data.loc[0, 'Theory_Vel'])
    print(f"T-stat results: {np.abs(tstat), p}\n")
    if p < 0.05:
        if np.abs(tstat) < abs(lowest_tstat):
            alpha = x
            lowest_tstat = tstat
            pvalue = p
print(alpha, lowest_tstat, pvalue)


