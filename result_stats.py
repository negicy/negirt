# 正規性の検定
from scipy import stats as st
import scipy.stats as stats

import seaborn as sns

headsData = pd.DataFrame({'top': top_acc_head, 'ours': ours_acc_head, 'random': random_acc_head})
tailData = pd.DataFrame({'top': top_acc_tail, 'ours': ours_acc_tail, 'random': random_acc_tail})
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

#print(scipy.stats.f_oneway(ours_acc_head, top_acc_head))
#print(scipy.stats.f_oneway(ours_acc_head, random_acc_head))

#print(scipy.stats.f_oneway(ours_acc_tail, top_acc_tail))
#print(scipy.stats.f_oneway(ours_acc_tail, random_acc_tail))
'''
anova_head = scipy.stats.f_oneway(ours_acc_head, top_acc_head, random_acc_head)
print(anova_head)
print(scipy.stats.f_oneway(ours_acc_tail, top_acc_tail, random_acc_tail))

print(scipy.stats.kruskal(ours_acc_tail, top_acc_tail, random_acc_tail))
print(scipy.stats.kruskal(ours_acc_head, top_acc_head, random_acc_head))
headsData = headsData.melt(var_name='groups', value_name='values')
tailData = tailData.melt(var_name='groups', value_name='values')
print(sp.posthoc_dscf(headsData, val_col='values', group_col='groups'))
print(sp.posthoc_dscf(tailData, val_col='values', group_col='groups'))
'''