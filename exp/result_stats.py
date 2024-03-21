# 正規性の検定
from scipy import stats as st
import scipy.stats as stats
import seaborn as sns
import pickle
import numpy as np

import pandas as pd
import scipy
filename = 'results/result_20231220_124432.pickle'
with open(filename, 'rb') as p:
    results = pickle.load(p)

AA_acc_allth = results['AA_acc_allth']
PI_acc_allth = results['PI_acc_allth']
ours_acc_allth = results['DI_acc_allth']

ours_acc_head = results['ours_acc_head']
ours_acc_tail = results['ours_acc_tail']
PI_acc_tail = results['PI_acc_tail']

AA_acc_head = results['AA_acc_head']
AA_acc_tail = results['AA_acc_tail']
print(AA_acc_allth)

AA_acc_0p7 = []
PI_acc_0p7 = []
ours_acc_0p7 = []
# AA_acc_allthの要素の各配列の[2]の平均値をAA_acc_0p7とする
for iter in range(len(AA_acc_allth)):
    AA_acc_0p7.append(AA_acc_allth[iter][2])
    PI_acc_0p7.append(PI_acc_allth[iter][2])
    ours_acc_0p7.append(ours_acc_allth[iter][2])


print(np.mean(ours_acc_tail), np.mean(AA_acc_tail))

#print(scipy.stats.f_oneway(ours_acc_head, top_acc_head))
#print(scipy.stats.f_oneway(ours_acc_head, random_acc_head))

#print(scipy.stats.f_oneway(ours_acc_tail, top_acc_tail))
#print(scipy.stats.f_oneway(ours_acc_tail, random_acc_tail))

print(len(AA_acc_head))
print(len(ours_acc_head))
print(len(AA_acc_tail))
print(len(ours_acc_tail))

tt_head = stats.ttest_ind(ours_acc_head, AA_acc_head, equal_var=False)
print(tt_head)
tt_tail = stats.ttest_ind(ours_acc_tail, AA_acc_tail, equal_var=False)
print(tt_tail)

print('===== PI(2PLM) vs AA @0.7 ======')
tt_PI_AA_0p7 = stats.ttest_ind(PI_acc_0p7, AA_acc_0p7, equal_var=False)
print(tt_PI_AA_0p7)

print('===== DI(2PLM) vs AA @0.7 ======')
tt_ours_AA_0p7 = stats.ttest_ind(ours_acc_0p7, AA_acc_0p7, equal_var=False)
print(tt_ours_AA_0p7)

print('===== PI(2PLM) vs AA @0.8 ======')
tt_tail = stats.ttest_ind(PI_acc_tail, AA_acc_tail, equal_var=False)
print(tt_tail)

print('===== DI(2PLM) vs AA @0.8 ======')
tt_ours_AA_0p8 = stats.ttest_ind(ours_acc_tail, AA_acc_tail, equal_var=False)
print(tt_ours_AA_0p8)


headsData = pd.DataFrame({'ours': ours_acc_head, 'AA': AA_acc_head})
tailData = pd.DataFrame({'ours': ours_acc_tail, 'AA': AA_acc_tail})
def check_norm():
    headsData = pd.DataFrame({'top': top_acc_head, 'ours': ours_acc_head, 'random': random_acc_head})
    tailData = pd.DataFrame({'top': top_acc_tail, 'ours': ours_acc_tail, 'random': random_acc_tail})

    print(headsData)
    print(st.bartlett(headsData['top'], headsData['ours'], headsData['random']))
    print(st.bartlett(tailData['top'], tailData['ours'], tailData['random']))

    print(st.shapiro(headsData))
    print(st.shapiro(tailData))

    print(st.shapiro(headsData['top']))
    print(st.shapiro(headsData['ours']))
    print(st.shapiro(headsData['random']))
    print(st.shapiro(tailData['top']))
    print(st.shapiro(tailData['ours']))
    print(st.shapiro(tailData['random']))


print(scipy.stats.kruskal(ours_acc_tail, top_acc_tail, random_acc_tail))
print(scipy.stats.kruskal(ours_acc_head, top_acc_head, random_acc_head))
headsData = headsData.melt(var_name='groups', value_name='values')
tailData = tailData.melt(var_name='groups', value_name='values')
print(sp.posthoc_dscf(headsData, val_col='values', group_col='groups'))
print(sp.posthoc_dscf(tailData, val_col='values', group_col='groups'))



