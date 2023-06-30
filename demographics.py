
import os
import pandas as pd
import numpy as np
pd.options.display.width = 0

input_dir = './../../datum/vfp/vfp/data/input/'
df = pd.read_excel(input_dir+'VFP_DeidentifiedDemographics.xlsx')


# UVFP age
vfp = df.iloc[:,:5]
vfp_age = vfp['Age:'].values
vfp_age_mean = np.round(np.mean(vfp_age),1)
vfp_age_sd = np.round(np.std(vfp_age),1)
print(f'mean age {vfp_age_mean} {vfp_age_sd}')
# UVFP sex
vfp_sex=  vfp['Sex:'].values
vfp_sex = [1 if x=='F' else 0 for x in vfp_sex]
female = np.sum(vfp_sex)
male = len(vfp_sex)-female

side = list(vfp['Side of Paralysis'].values)
side = [n.lower() for n in side]
left = side.count('left vf paralysis')
right = side.count('right vf paralysis')
print(f'left {left} right {right}')

# Controls
controls= df.iloc[:,5:]
controls_age = controls['Age:.1'].values
controls_age_mean = np.round(np.mean(controls_age),1)
controls_age_sd = np.round(np.std(controls_age),1)
print(f'mean age {controls_age_mean} +- {controls_age_sd}')
controls_sex=  controls['Sex:.1'].values
controls_sex= [1 if x=='F' else 0 for x in controls_sex]
female = np.sum(controls_sex)
male = len(controls_sex)-female


age = np.concatenate([vfp_age, controls_age])
age_mean = np.round(np.mean(age),1)
age_sd = np.round(np.std(age),1)
print(f'mean age {age_mean} {age_sd}')

