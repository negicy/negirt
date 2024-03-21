# 正規性の検定
from scipy import stats as st
import scipy.stats as stats
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import scipy
import scikit_posthocs as sp

filename = 'results/result_20231220_124432.pickle'
with open(filename, 'rb') as p:
    results = pickle.load(p)

AA_acc_allth = results['AA_acc_allth']
PI_acc_allth = results['PI_acc_allth']
ours_acc_allth = results['DI_acc_allth']
top_acc_allth = results['top_acc_allth']
random_acc_allth = results['random_acc_allth']

ours_acc_head = results['ours_acc_head']
ours_acc_tail = results['ours_acc_tail']
PI_acc_tail = results['PI_acc_tail']

AA_acc_head = results['AA_acc_head']
AA_acc_tail = results['AA_acc_tail']

PI_onepl_acc_allth = results['PI_onepl_acc_allth']
ours_onepl_acc_allth = results['DI_onepl_acc_allth']

ours_acc_head = results['ours_acc_head']
ours_acc_tail = results['ours_acc_tail']
PI_acc_tail = results['PI_acc_tail']

AA_acc_head = results['AA_acc_head']
AA_acc_tail = results['AA_acc_tail']

print(AA_acc_allth)

AA_acc_0p7 = []
PI_acc_0p7 = []
ours_acc_0p7 = []
top_acc_0p7 = []
random_acc_0p7 = []

AA_acc_0p8 = []
PI_acc_0p8 = []
ours_acc_0p8 = []
top_acc_0p8 = []
random_acc_0p8 = []

PI_onepl_acc_0p7 = []
ours_onepl_acc_0p7 = []

PI_onepl_acc_0p8 = []
ours_onepl_acc_0p8 = []


# AA_acc_allthの要素の各配列の[2]の平均値をAA_acc_0p7とする
for iter in range(len(AA_acc_allth)):
    AA_acc_0p7.append(AA_acc_allth[iter][2])
    PI_acc_0p7.append(PI_acc_allth[iter][2])
    ours_acc_0p7.append(ours_acc_allth[iter][2])
    top_acc_0p7.append(top_acc_allth[iter][2])
    random_acc_0p7.append(random_acc_allth[iter][2])

    AA_acc_0p8.append(AA_acc_allth[iter][3])
    PI_acc_0p8.append(PI_acc_allth[iter][3])
    ours_acc_0p8.append(ours_acc_allth[iter][3])
    top_acc_0p8.append(top_acc_allth[iter][3])
    random_acc_0p8.append(random_acc_allth[iter][3])

    PI_onepl_acc_0p7.append(PI_onepl_acc_allth[iter][2])
    ours_onepl_acc_0p7.append(ours_onepl_acc_allth[iter][2])

    PI_onepl_acc_0p8.append(PI_onepl_acc_allth[iter][3])
    ours_onepl_acc_0p8.append(ours_onepl_acc_allth[iter][3])


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

def check_norm(data):
    # headsData = pd.DataFrame({'top': top_acc_0p7, 'ours': ours_acc_0p7, 'random': random_acc_0p7})

    print(data)
    # 等分散性の検定
    print('===== 等分散性の検定 =====')
    # print(st.bartlett(data['top'], data['ours'], data['PI'], data['AA'],data['random']))
    print(st.bartlett(data['ours'], data['PI'], data['AA'],data['random']))
    # print(st.bartlett(tailData['top'], tailData['ours'], tailData['random']))

    # 正規性の検定
    print("===== 正規性の検定 =====")
    # print(st.shapiro(data))
    # print(st.shapiro(tailData))
    # print(st.shapiro(data['top']))
    print(st.shapiro(data['ours']))
    print(st.shapiro(data['AA']))
    print(st.shapiro(data['PI']))
    print(st.shapiro(data['random']))
    print('=====================')

    #print(st.shapiro(tailData['top']))
    #print(st.shapiro(tailData['ours']))
    #print(st.shapiro(tailData['random']))

Data_0p7 = pd.DataFrame({'ours': ours_acc_0p7, 'PI':PI_acc_0p7, 'AA': AA_acc_0p7, 'random': random_acc_0p7})
Data_0p8 = pd.DataFrame({'ours': ours_acc_0p8, 'PI':PI_acc_0p8, 'AA': AA_acc_0p8, 'random': random_acc_0p8})
Data_onepl_0p7 = pd.DataFrame({'ours': ours_onepl_acc_0p7, 'PI':PI_onepl_acc_0p7, 'AA': AA_acc_0p7, 'random': random_acc_0p7})
Data_onepl_0p8 = pd.DataFrame({'ours': ours_onepl_acc_0p8, 'PI':PI_onepl_acc_0p8, 'AA': AA_acc_0p8, 'random': random_acc_0p8})
check_norm(Data_0p7)
check_norm(Data_0p8)


# Data_0p7 = Data_0p7.melt(var_name='groups', value_name='values')

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# print(sp.posthoc_dscf(Data_0p7, val_col='values', group_col='groups'))
print(scipy.stats.f_oneway(ours_acc_0p7, PI_acc_0p7, random_acc_0p7, AA_acc_0p7))
print(scipy.stats.f_oneway(ours_acc_0p7, PI_acc_0p7, AA_acc_0p7))

Data_0p7 = pd.DataFrame({'ours': ours_acc_0p7, 'PI':PI_acc_0p7, 'AA': AA_acc_0p7, 'random': random_acc_0p7})

group = ['DI' for _ in range(len(ours_acc_0p7))] + ['PI' for _ in range(len(PI_acc_0p7))] + \
        ['AA' for _ in range(len(AA_acc_0p7))] + ['random' for _ in range(len(random_acc_0p7))]

df1 = pd.DataFrame({'group': group, 'score': pd.concat([Data_0p7['ours'], Data_0p7['PI'], Data_0p7['AA'], Data_0p7['random']])})

print(pairwise_tukeyhsd(df1.score, df1.group).summary())


# onepl 0.7
print(scipy.stats.f_oneway(ours_acc_0p8, PI_acc_0p8, random_acc_0p8, AA_acc_0p8))
print(scipy.stats.f_oneway(ours_acc_0p8, PI_acc_0p8, AA_acc_0p8))

Data_0p8 = pd.DataFrame({'ours': ours_acc_0p8, 'PI':PI_acc_0p8, 'AA': AA_acc_0p8, 'random': random_acc_0p8})

group = ['DI' for _ in range(len(ours_acc_0p8))] + ['PI' for _ in range(len(PI_acc_0p8))] + \
        ['AA' for _ in range(len(AA_acc_0p8))] + ['random' for _ in range(len(random_acc_0p8))]

df2 = pd.DataFrame({'group': group, 'score': pd.concat([Data_0p8['ours'], Data_0p8['PI'], Data_0p8['AA'], Data_0p8['random']])})

print(pairwise_tukeyhsd(df2.score, df2.group).summary())


check_norm(Data_onepl_0p7)

print(scipy.stats.f_oneway(ours_onepl_acc_0p7, PI_onepl_acc_0p7, random_acc_0p7, AA_acc_0p7))
f_stats, p_value = scipy.stats.f_oneway(ours_onepl_acc_0p7, PI_onepl_acc_0p7, random_acc_0p7, AA_acc_0p7)
print("F値:", f_stats)

Data_onepl_0p7 = pd.DataFrame({'ours': ours_onepl_acc_0p7, 'PI':PI_onepl_acc_0p7, 'AA': AA_acc_0p7, 'random': random_acc_0p7})

group = ['DI' for _ in range(len(ours_onepl_acc_0p7))] + ['PI' for _ in range(len(PI_onepl_acc_0p7))] + \
        ['AA' for _ in range(len(AA_acc_0p7))] + ['random' for _ in range(len(random_acc_0p7))]

df1 = pd.DataFrame({'group': group, 'score': pd.concat([Data_onepl_0p7['ours'], Data_onepl_0p7['PI'], Data_onepl_0p7['AA'], Data_onepl_0p7['random']])})

print(pairwise_tukeyhsd(df1.score, df1.group).summary())

# onepl 0.8
check_norm(Data_onepl_0p7)

print(scipy.stats.f_oneway(ours_onepl_acc_0p8, PI_onepl_acc_0p8, random_acc_0p8, AA_acc_0p8))
print(scipy.stats.f_oneway(ours_onepl_acc_0p8, PI_onepl_acc_0p8, AA_acc_0p8))

Data_onepl_0p8 = pd.DataFrame({'ours': ours_onepl_acc_0p8, 'PI':PI_onepl_acc_0p8, 'AA': AA_acc_0p8, 'random': random_acc_0p8})

group = ['DI' for _ in range(len(ours_onepl_acc_0p8))] + ['PI' for _ in range(len(PI_onepl_acc_0p8))] + \
        ['AA' for _ in range(len(AA_acc_0p8))] + ['random' for _ in range(len(random_acc_0p8))]

df1 = pd.DataFrame({'group': group, 'score': pd.concat([Data_onepl_0p8['ours'], Data_onepl_0p8['PI'], Data_onepl_0p8['AA'], Data_onepl_0p8['random']])})

print(pairwise_tukeyhsd(df1.score, df1.group).summary())