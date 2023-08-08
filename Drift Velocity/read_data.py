
import pandas as pd
import numpy as np

# with pd.read_excel('alpha_MaxC_60000_Grad_0.000405_raw_data.xlsx') as data:
#     print(data)

file = 'pos_10_alpha_4250_to_4450_MaxC_60000_Grad_0.000405.xlsx'
data = pd.read_excel(file)
print(data.head())
