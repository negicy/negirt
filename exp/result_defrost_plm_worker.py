import pickle
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy
import scikit_posthocs as sp
from assignment_method import *
from irt_method import *
from simulation import *
from make_candidate import *
import japanize_matplotlib

filename = 'results/result_20231220_124432.pickle'

with open(filename, 'rb') as p:
    results = pickle.load(p)

welldone_dist = results['welldone_dist']
iteration_time = 200

ours_acc = results['ours_acc']
ours_var = results['ours_var']
ours_tp = results['ours_tp']
ours_acc_std = results['ours_acc_std']
ours_var_std = results['ours_var_std']

DI_onepl_acc = results['DI_onepl_acc']
DI_onepl_var = results['DI_onepl_var']
DI_onepl_tp = results['DI_onepl_tp']
DI_onepl_acc_std = results['DI_onepl_acc_std']

DI_onepl_margin_acc = results['DI_onepl_margin_acc']
DI_onepl_margin_var = results['DI_onepl_margin_var']
DI_onepl_margin_tp = results['DI_onepl_margin_tp']
DI_onepl_margin_acc_std = results['DI_onepl_margin_acc_std']

PI_onepl_acc = results['PI_onepl_acc']
PI_onepl_var = results['PI_onepl_var']
PI_onepl_tp = results['PI_onepl_tp']

PI_onepl_margin_acc = results['PI_onepl_margin_acc']
PI_onepl_margin_var = results['PI_onepl_margin_var']
PI_onepl_margin_tp = results['PI_onepl_margin_tp']
PI_onepl_margin_acc_std = results['PI_onepl_margin_acc_std']
PI_onepl_margin_var_std = results['PI_onepl_margin_var_std']

PI_onepl_acc = results['PI_onepl_acc']
PI_onepl_var = results['PI_onepl_var']
PI_onepl_tp = results['PI_onepl_tp']
PI_onepl_acc_std = results['PI_onepl_acc_std']
PI_onepl_var_std = results['PI_onepl_var_std']

top_acc = results['top_acc']
top_var = results['top_var']
top_tp = results['top_tp']
top_acc_std = results['top_acc_std']
top_var_std = results['top_var_std']

AA_acc = results['AA_acc']
AA_var = results['AA_var']
AA_tp = results['AA_tp']
AA_acc_std = results['AA_acc_std']
AA_var_std = results['AA_var_std']

random_acc = results['random_acc']
random_var = results['random_var']
random_tp = results['random_tp']
random_acc_std = results['random_acc_std']
random_var_std = results['random_var_std']

PI_acc = results['PI_acc']
PI_var = results['PI_var']
PI_tp = results['PI_tp']
PI_acc_std = results['PI_acc_std']
PI_var_std = results['PI_var_std']

PI_all_assign_dic_alliter = results['PI_all_assign_dic_alliter']

PI_twopl_res_dic = results['PI_twopl_res_dic']
DI_twopl_res_dic = results['DI_twopl_res_dic']

PI_onepl_res_dic = results['PI_onepl_res_dic']
DI_onepl_res_dic = results['DI_onepl_res_dic']

PI_onepl_margin_res_dic = results['PI_onepl_margin_res_dic']
DI_onepl_margin_res_dic = results['DI_onepl_margin_res_dic']

worker_rank_dict_PI = results['worker_rank_dict_PI']
worker_rank_dict_DI = results['worker_rank_dict_DI'] 

worker_rank_dict_PI_onepl = results['worker_rank_dict_PI_onepl']
worker_rank_dict_DI_onepl = results['worker_rank_dict_DI_onepl'] 

worker_rank_dict_PI_onepl_margin = results['worker_rank_dict_PI_onepl_margin']
worker_rank_dict_DI_onepl_margin = results['worker_rank_dict_DI_onepl_margin']

threshold = list([i / 10 for i in range(50, 81)])
threshold=[0.5, 0.6, 0.7, 0.8]

for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

plt.rcParams["font.size"] = 22
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.4, color='red')
# plt.show()

DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ot_task.append(DI_onepl_res_dic[th][0])
    PI_ot_task.append(PI_onepl_res_dic[th][0])

    DI_ut_task.append(DI_onepl_res_dic[th][1])
    PI_ut_task.append(PI_onepl_res_dic[th][1])

    PI_of_task.append(PI_onepl_res_dic[th][2])
    DI_of_task.append(DI_onepl_res_dic[th][2])

    PI_uf_task.append(DI_onepl_res_dic[th][3])
    DI_uf_task.append(DI_onepl_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_of_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_of_task, width, bottom=PI_ot_task, color='violet')


plt.ylabel('Number of tasks')
plt.title('Number of good assignment  by DI,PI(by 2PLM)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

# タスクのあるワーカのヒストグラム
# PIの，worker_with_tasksを取り出す: PI_all_assign_dic_alliterから
worker_with_task = results['worker_with_task']
worker_with_task['PI'] = {}
print(PI_all_assign_dic_alliter[0.5][0])
for th in threshold:
  worker_with_task['PI'][th] = 0
  PI_all_assign_dic = PI_all_assign_dic_alliter[th]
  worker_num_perth = 0
  for i in range(0, iteration_time):
    for w in PI_all_assign_dic[i][0].values():
      if w in worker_with_task:
        # worker_with_task[w] += 1
        continue
      else:
        # worker_with_task[w] = 1
        worker_num_perth += 1
  worker_with_task['PI'][th] = worker_num_perth
   

num_worker = [[], [], []]
for th in [0.5, 0.6, 0.7, 0.8]:
  num_worker[0].append(worker_with_task['PI'][th] / iteration_time)
  num_worker[1].append(worker_with_task['ours'][th] / iteration_time)
  num_worker[2].append(worker_with_task['AA'][th] / iteration_time)
w = 0.4
y_PI = num_worker[0]
y_DI = num_worker[1]
y_AA = num_worker[2]

x1 = [1, 2, 3, 4]
x2 = [1.3, 2.3, 3.3, 4.3]


print(f'random acc: {random_acc}')
print(f'top acc: {top_acc}')
print(f'AA acc: {AA_acc}')
print(f'ours acc: {ours_acc}')
print(f'PI acc: {PI_acc}')
# print(f'PI margin acc: {PI_onepl_margin_acc}')
# print(f'DI margin acc: {DI_onepl_margin_acc}')
print(f'PI_onepl_acc: {PI_onepl_acc}')
print(f'DI_onepl_acc: {DI_onepl_acc}')


PI_twopl_res_dic = results['PI_twopl_res_dic']
DI_twopl_res_dic = results['DI_twopl_res_dic']

PI_onepl_res_dic = results['PI_onepl_res_dic']
DI_onepl_res_dic = results['DI_onepl_res_dic']

PI_onepl_margin_res_dic = results['PI_onepl_margin_res_dic']
DI_onepl_margin_res_dic = results['DI_onepl_margin_res_dic']

worker_rank_dict_PI = results['worker_rank_dict_PI']
worker_rank_dict_DI = results['worker_rank_dict_DI'] 

worker_rank_dict_PI_onepl = results['worker_rank_dict_PI_onepl']
worker_rank_dict_DI_onepl = results['worker_rank_dict_DI_onepl'] 

worker_rank_dict_PI_onepl_margin = results['worker_rank_dict_PI_onepl_margin']
worker_rank_dict_DI_onepl_margin = results['worker_rank_dict_DI_onepl_margin']


'''

print(f"PI variance:{PI_onepl_var}")
for th in threshold:
    ind = np.array(worker_rank_dict_PI_onepl[th].keys())

    assigned_ot_ut = worker_rank_dict_PI_onepl[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    plt.bar(keys, ot_values, color='violet', label='good')
    plt.bar(keys, of_values, color='violet', bottom=ot_values)
    #plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    #plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xticks([])
    plt.xlabel('Workers sorted by skill')
    plt.ylabel('Assigned tasks')
    plt.ylim(0,10)
    plt.legend()

    # Show the plot
    plt.show()

print(f"DI variance:{DI_onepl_var}")
for th in threshold:

    ind = np.array(worker_rank_dict_DI_onepl[th].keys())

    assigned_ot_ut = worker_rank_dict_DI_onepl[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    # 積み上げグラフとして表示：4段
    plt.bar(keys, ot_values, color='red', label='ot')
    plt.bar(keys, ut_values, color='orange', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')


    # Labels and title
    plt.xticks([])
    plt.xlabel('Workers sorted by skill')
    plt.ylabel('Assigned tasks')
    plt.ylim(0,10)
    plt.legend()

    # Show the plot
    plt.show()

'''
    
print(f"PI variance:{PI_var}")
for th in threshold:
    print(PI_var)
    ind = np.array(worker_rank_dict_PI[th].keys())

    assigned_ot_ut = worker_rank_dict_PI[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting

    plt.bar(keys, ot_values, color='purple', label='PIによるgood割当て')
    plt.bar(keys, of_values, color='purple', bottom=ot_values)

    #plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    #plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xticks([])
    plt.xlabel('スキルの高い順に並べたワーカ')
    plt.ylabel('割\n当\nて\nら\nれ\nた\nタ\nス\nク\n数',rotation=0,labelpad=15,va='center')
    # plt.title(f'Stacked Histogram of OT and UT values(PI-2PLM@{th})')
    plt.ylim(0,7)
    plt.legend()

    # Show the plot
    plt.show()

print(f"DI variance:{ours_var}")
for th in threshold:

    ind = np.array(worker_rank_dict_DI[th].keys())

    assigned_ot_ut = worker_rank_dict_DI[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    # 積み上げグラフとして表示：4段
    plt.bar(keys, ot_values, color='red', label='DIによるgood割当て')
    plt.bar(keys, of_values, color='red', bottom=ot_values)

    # Labels and title
    plt.xticks([])
    plt.xlabel('スキルの高い順に並べたワーカ')
    plt.ylabel('割\n当\nて\nら\nれ\nた\nタ\nス\nク\n数',rotation=0,labelpad=15,va='center')
    #plt.title(f'Stacked Histogram of OT and UT values(DI-2PLM@{th})')
    plt.ylim(0,7)
    plt.legend()
    # Show the plot
    plt.show()