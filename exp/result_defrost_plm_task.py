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

filename = 'results/result_20231214_210938.pickle'

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


test_task_size = 60
# ヒストグラム描画: 横軸: threshold, 縦軸: θ < bで正答したタスク数
# パラメータの関係と正誤の関係 1PLM
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

PI_ot_task_onepl = []
PI_ot_task_twopl = []

for th in threshold:
    #print(PI_onepl_res_dic[th])
    o_size_onepl = PI_onepl_res_dic[th][0] + PI_onepl_res_dic[th][2]
    o_size_twopl = PI_twopl_res_dic[th][0] + PI_twopl_res_dic[th][2]
    PI_ot_task_onepl.append(PI_onepl_res_dic[th][0]/o_size_onepl)
    PI_ot_task_twopl.append(PI_twopl_res_dic[th][0]/o_size_twopl)
    print(PI_twopl_res_dic[th][0]/o_size_twopl)


ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, PI_ot_task_onepl, width, color='violet', label='PI(1PLM)')
#p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p2 = ax.bar(ind + width/2, PI_ot_task_twopl, width,  color='purple', label='PI(2PLM)')
#p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')

plt.ylabel('正解率')
#plt.title('PIによる正解タスク数')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
# plt.yticks(np.arange(0, 1))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
# 右下に凡例を表示
plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=18)
#plt.legend()
plt.show()

'''

# パラメータの関係と正誤の関係 1PLM+Margin
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ot_task.append(DI_onepl_margin_res_dic[th][0])
    PI_ot_task.append(PI_onepl_margin_res_dic[th][0])

    DI_ut_task.append(DI_onepl_margin_res_dic[th][1])
    PI_ut_task.append(PI_onepl_margin_res_dic[th][1])

    PI_of_task.append(PI_onepl_margin_res_dic[th][2])
    DI_of_task.append(DI_onepl_margin_res_dic[th][2])

    PI_uf_task.append(DI_onepl_margin_res_dic[th][3])
    DI_uf_task.append(DI_onepl_margin_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')
#p1 = ax.bar(ind - width/2, PI_ot_task, width, color='red')
#p3 = ax.bar(ind + width/2, PI_of_task, width,  color='purple')
plt.ylabel('Number of tasks')
plt.title('Number of correctly answered task by DI,PI(1PLM+Margin)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

# パラメータの関係と正誤の関係 2PLM
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ot_task.append(DI_twopl_res_dic[th][0])
    PI_ot_task.append(PI_twopl_res_dic[th][0])

    DI_ut_task.append(DI_twopl_res_dic[th][1])
    PI_ut_task.append(PI_twopl_res_dic[th][1])

    PI_of_task.append(PI_twopl_res_dic[th][2])
    DI_of_task.append(DI_twopl_res_dic[th][2])

    PI_uf_task.append(DI_twopl_res_dic[th][3])
    DI_uf_task.append(DI_twopl_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')

plt.ylabel('Number of tasks')
plt.title('Number of correctly answered task by DI,PI(2PLM)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

'''