import numpy as np
import pandas as pd
from scipy import stats

# The length of time to import the data into a df takes around 40s.
Diffusion_Data = pd.read_excel("1.000_to_1.200_Diff_Conts_30001_DataPts_1000_Time.xlsx")
pd.set_option('display.max_columns', None)  # Show all the columns

# Get the column titles for use in pandas functions
pos_diff = Diffusion_Data.columns[-1]
diff_const = Diffusion_Data.columns.values[2]
# Diffusion_Data2 will have outliers filtered out, while Diffusion_Data is unfiltered.
Diffusion_Data2 = Diffusion_Data

describe_diff_by_const = Diffusion_Data2.groupby(diff_const)[pos_diff].describe()
# Before removing outliers get a count of the number of datapoints
count = describe_diff_by_const.iloc[0, 0]

# Filter out datapoints that are greater than 3xSD
filtered_pos = pd.Series()
for x in describe_diff_by_const.index:
    alpha_pos_val = Diffusion_Data2[pos_diff].loc[Diffusion_Data2['R_T_Diff_Con'] == x]
    alpha_pos_val = alpha_pos_val.loc[stats.zscore(np.abs(alpha_pos_val)) <= 3]
    filtered_pos = pd.concat([filtered_pos, alpha_pos_val])
Diffusion_Data2 = Diffusion_Data2[Diffusion_Data2[pos_diff].isin(filtered_pos)]

# 95% of sample means will be expected to lie within a confidence interval of ±1.96 standard errors of the sample mean
describe_diff_by_const['std_error'] = describe_diff_by_const['std']/np.sqrt(describe_diff_by_const['count'])
describe_diff_by_const['95%_conf_int'] = describe_diff_by_const['std_error'] * 1.96
# Filter out the means that are not within the 95% confidence interval of the standard error.
describe_diff_by_const = describe_diff_by_const[np.abs(describe_diff_by_const['mean']) < describe_diff_by_const['95%_conf_int']]

# Get a list of constants that are within in the interval.
describe_diff_by_const_list = describe_diff_by_const.reset_index().iloc[:, 0]
# Now filter out those constants that are not in the dataframe by only selecting constants in the list.
Diffusion_Data2 = Diffusion_Data2[Diffusion_Data2[diff_const].isin(describe_diff_by_const_list)]

# All outliers greater than 3*std, and means that are not within the 95% confidence interval of the standard error have been filtered out.
describe_diff_by_const = Diffusion_Data2.groupby(diff_const)[pos_diff].describe()
describe_diff_by_const['3*std'] = describe_diff_by_const['std']*3 # New value after outliers removed.
describe_diff_by_const['theor_std_error'] = Diffusion_Data['Std_Dev'][0]/np.sqrt(count) # now use "count" calculated before datapoints were filtered.
describe_diff_by_const['std_error'] = describe_diff_by_const['std']/np.sqrt(describe_diff_by_const['count'])
describe_diff_by_const['95%_conf_int'] = describe_diff_by_const['std_error'] * 1.96
# Calculate the Bhattacharyya Distance between the calculated and theoretical normal distributions to see which is the closest fit.
describe_diff_by_const['bhattacharyya'] = 0.25*((describe_diff_by_const['mean'] - 0)**2)/(describe_diff_by_const['std']**2 + (Diffusion_Data['Std_Dev'][0])**2) + 0.5*np.log(((describe_diff_by_const['std'])**2 + (Diffusion_Data['Std_Dev'][0])**2)/(2*(describe_diff_by_const['std'])*(Diffusion_Data['Std_Dev'][0])))
describe_diff_by_const['t-test'] = [(stats.ttest_1samp(Diffusion_Data2[Diffusion_Data2[diff_const] == x][pos_diff], popmean=0)) for x in describe_diff_by_const.index.values]
describe_diff_by_const['shapiro'] = [stats.shapiro(Diffusion_Data2[Diffusion_Data2[diff_const] == x][pos_diff]) for x in describe_diff_by_const.index.values]
print(describe_diff_by_const.sort_values(by='bhattacharyya'))