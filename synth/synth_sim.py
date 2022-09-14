import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import girth
import random
import scipy
import scikit_posthocs as sp
from assignment_method import *
from irt_method import *
from simulation import *
# from survey import *ttrr333232

# 1. worker list, task listをsynthデータで代替する
# 2. IRTによるパラメータ推定は省略

import numpy as np

from girth.synthetic import create_synthetic_irt_dichotomous
from girth import onepl_mml
import matplotlib.pyplot as plt 
from scipy.stats import norm
import matplotlib.pyplot as plt 

worker_size = 100
task_size = 200

user_param_list = norm.rvs(loc=1.2, scale=0.25, size=worker_size)
item_param_list = norm.rvs(loc=0.8, scale=0.25, size=task_size)

bins=np.linspace(-3, 3, 20)
plt.hist([user_param_list, item_param_list], bins, label=['worker', 'task'])

a_list = np.array([1]*task_size)
user_param = {}
item_param = {}

for i in range(0, worker_size):
    worker_id = 'w' + str(i+1)
    user_param[worker_id] = user_param_list[i]

for j in range(0, task_size):
    task_id = 't' + str(j+1)
    item_param[task_id] = item_param_list[j]



worker_list = list(user_param.keys())
task_list = list(item_param.keys())

# 仮想項目反応データ生成
input_data = create_synthetic_irt_dichotomous(item_param_list, a_list, user_param_list)
input_df = pd.DataFrame(input_data, columns=worker_list, index=task_list)

qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
  qualify_dic[qt] = list(input_df.T[qt])
q_data = np.array(list(qualify_dic.values()))
params = run_girth_rasch(q_data, task_list, worker_list)
full_item_param = params[0]
full_user_param = params[1]



ours_acc_allth = []
ours_var_allth = []

top_acc_allth = []
top_var_allth = []

AA_acc_allth = []
AA_var_allth = []

random_acc_allth = []
random_var_allth = []

full_irt_acc_allth = []
full_irt_var_allth = []

threshold = list([i / 100 for i in range(50, 81)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)

ours_output_alliter = {}
full_output_alliter = {}

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 5
worker_with_task = {'ours': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}, 'AA': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}}
for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []

  top_acc_perth = []
  top_var_perth = []

  AA_acc_perth = []
  AA_var_perth = []

  random_acc_perth = []
  random_var_perth = []

  full_irt_acc_perth = []
  full_irt_var_perth = []
  
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']

  # 各手法でのワーカ候補作成
  ours_output =  make_candidate(threshold, input_df, worker_list, test_worker, qualify_task, test_task, full_item_param)
  ours_candidate = ours_output[0]
  est_user_param = ours_output[1]

  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  AA_candidate = AA_assignment(threshold, input_df, test_worker, qualify_task, test_task)

  full_output = make_candidate_all(threshold, input_df, task_list, worker_list, test_worker, test_task, full_user_param, full_item_param)
  full_irt_candidate = full_output

  # 保存用
  ours_output_alliter[iteration] = ours_output
  full_output_alliter[iteration] = full_output

  # worker_c_th = {th: {task: [workers]}}
  for th in ours_candidate:
    candidate_dic = ours_candidate[th]
    
    assign_dic_opt = {}
    assigned = optim_assignment(candidate_dic, test_worker, test_task, est_user_param)

    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    # print('th'+str(th)+'assignment size'+str(len(assign_dic_opt)))
    # print(len(assign_dic_opt.keys()))
    if th in [0.5, 0.6, 0.7, 0.8]:
      welldone_dist[th] += welldone_count(th, assign_dic_opt, user_param, item_param) / len(test_task) 
      # 割当て候補人数を数える
      worker_with_task_list = []
      for worker_set in ours_candidate[th].values():
        for worker in worker_set:
          if worker not in worker_with_task_list:
            worker_with_task_list.append(worker)
      
      worker_with_task['ours'][th] += len(worker_with_task_list)
  
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    var = task_variance(assign_dic_opt, test_worker)
    
    ours_acc_perth.append(acc)
    ours_var_perth.append(var)
  
  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)
 
  
  for candidate_dic in top_candidate.values():

    index = 0
    assign_dic = {}
    for task in candidate_dic:
      index = index % 5
      candidate_list = candidate_dic[task]
      assign_dic[task] = random.choice(candidate_list)
      index += 1
   
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    top_acc_perth.append(acc)
    top_var_perth.append(var)

  top_acc_allth.append(top_acc_perth)
  top_var_allth.append(top_var_perth)

  for th in AA_candidate:
    candidate_dic = AA_candidate[th]
    
    assign_dic_opt = {}
    assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)
    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    
   
      

    if th in [0.5, 0.6, 0.7, 0.8]:
      welldone_dist[th] += welldone_count(th, assign_dic_opt, user_param, item_param) / len(test_task) 
      # 割当て候補人数を数える
      # worker_with_task['AA'] += len(assign_dic_opt[th])
      worker_with_task_list = []
      for worker_set in AA_candidate[th].values():
        for worker in worker_set:
          if worker not in worker_with_task_list:
            worker_with_task_list.append(worker)

      worker_with_task['AA'][th] += len(worker_with_task_list)

     # 割り当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    var = task_variance(assign_dic_opt, test_worker)

    AA_acc_perth.append(acc)
    AA_var_perth.append(var)

  AA_acc_allth.append(AA_acc_perth)
  AA_var_allth.append(AA_var_perth)
 
  for th in full_irt_candidate:
    candidate_dic = full_irt_candidate[th]
    assign_dic_opt = {}
    # assign_dic = assignment(candidate_dic, test_worker)
    assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)

    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    # assign_dic = assignment(candidate_dic, test_worker)
    print('==========================================-------')
    print(th, assign_dic_opt)
    # 割当て結果の精度を求める

    acc = accuracy(assign_dic_opt, input_df)
    var = task_variance(assign_dic_opt, test_worker)
    
    # 割当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    var = task_variance(assign_dic_opt, test_worker)
    
    full_irt_acc_perth.append(acc)
    full_irt_var_perth.append(var)

  full_irt_acc_allth.append(full_irt_acc_perth)
  full_irt_var_allth.append(full_irt_var_perth)
  
  for th in range(0, len(threshold)):
    assign_dic = random_assignment(test_task, test_worker)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    random_acc_perth.append(acc)
    random_var_perth.append(var)

  random_acc_allth.append(random_acc_perth)
  random_var_allth.append(random_var_perth)

ours_acc = [0] * len(threshold)
ours_var = [0] * len(threshold)
ours_acc_std =  []
ours_var_std =  []

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)
top_acc_std = []
top_var_std = []

AA_acc = [0] * len(threshold)
AA_var = [0] * len(threshold)
AA_acc_std = []
AA_var_std = []

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)
random_acc_std = []
random_var_std = []

full_irt_acc = [0] * len(threshold)
full_irt_var = [0] * len(threshold)
full_acc_std = []
full_var_std = []

for th in range(0, len(threshold)):
  ours_acc_sum = 0
  ours_var_sum = 0
  ours_acc_num = 0
  ours_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    # 
    if ours_acc_allth[i][th] != "null":
      list_acc_th.append(ours_acc_allth[i][th])
      ours_acc_sum += ours_acc_allth[i][th]
      ours_acc_num += 1
    ours_var_sum += ours_var_allth[i][th]
    list_var_th.append(ours_var_allth[i][th])
    ours_var_num += 1
  # acc, var の平均を計算

  ours_acc[th] = ours_acc_sum / ours_acc_num
  ours_var[th] = ours_var_sum / ours_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  ours_acc_std.append(acc_std)
  ours_var_std.append(var_std)

  if th == 0:
    ours_acc_head = list_acc_th
  if th == len(threshold)-1:
    ours_acc_tail = list_acc_th
  
for th in range(0, len(threshold)):
  top_acc_sum = 0
  top_var_sum = 0
  top_acc_num = 0
  top_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if top_acc_allth[i][th] != "null":
      top_acc_sum += top_acc_allth[i][th]
      list_acc_th.append(top_acc_allth[i][th])
      top_acc_num += 1
    if top_var_allth[i][th] != "null":
      top_var_sum += top_var_allth[i][th]
      list_var_th.append(top_var_allth[i][th])
      top_var_num += 1
    
  top_acc[th] = top_acc_sum / top_acc_num
  top_var[th] = top_var_sum / top_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  top_acc_std.append(acc_std)
  top_var_std.append(var_std)
  if th == 0:
    top_acc_head = list_acc_th
  if th == len(threshold)-1:
    top_acc_tail = list_acc_th

for th in range(0, len(threshold)):
  AA_acc_sum = 0
  AA_var_sum = 0
  AA_acc_num = 0
  AA_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  
  for i in range(0, iteration_time):
    #
    if AA_acc_allth[i][th] != "null":
      AA_acc_sum += AA_acc_allth[i][th]
      list_acc_th.append(AA_acc_allth[i][th])
      AA_acc_num += 1
  
    if AA_var_allth[i][th] != "null":
      AA_var_sum += AA_var_allth[i][th]
      list_var_th.append(AA_var_allth[i][th])
      AA_var_num += 1
  # print(AA_acc_allth)
  AA_acc[th] = AA_acc_sum / AA_acc_num
  AA_var[th] = AA_var_sum / AA_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  AA_acc_std.append(acc_std)
  AA_var_std.append(var_std)
  if th == 0:
    AA_acc_head = list_acc_th
  if th == len(threshold)-1:
    AA_acc_tail = list_acc_th
  
for th in range(0, len(threshold)):
  random_acc_sum = 0
  random_var_sum = 0
  random_acc_num = 0
  random_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    # e
    if random_acc_allth[i][th] != "null":
      random_acc_sum += random_acc_allth[i][th]
      list_acc_th.append(random_acc_allth[i][th])
      random_acc_num += 1
    if random_var_allth[i][th] != "null":
      random_var_sum += random_var_allth[i][th]
      list_var_th.append(random_var_allth[i][th])
      random_var_num += 1
    
  random_acc[th] = random_acc_sum / random_acc_num
  random_var[th] = random_var_sum / random_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  random_acc_std.append(acc_std)
  random_var_std.append(var_std)

  if th == 0:
    random_acc_head = list_acc_th
  if th == len(threshold)-1:
    random_acc_tail = list_acc_th


for th in range(0, len(threshold)):
  full_irt_acc_sum = 0
  full_irt_var_sum = 0
  full_irt_acc_num = 0
  full_irt_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if full_irt_acc_allth[i][th] != "null":
      full_irt_acc_sum += full_irt_acc_allth[i][th]
      list_acc_th.append(random_acc_allth[i][th])
      full_irt_acc_num += 1
    if full_irt_var_allth[i][th] != "null":
      full_irt_var_sum += full_irt_var_allth[i][th]
      list_var_th.append(full_irt_var_allth[i][th])
      full_irt_var_num += 1
    
  full_irt_acc[th] = full_irt_acc_sum / full_irt_acc_num
  full_irt_var[th] = full_irt_var_sum / full_irt_var_num
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  full_acc_std.append(acc_std)
  full_var_std.append(var_std)


# 割当て結果保存:
import datetime
now = datetime.datetime.now()
result = {
  'ours_output': ours_output_alliter, 'full_output': full_output_alliter, 
  'ours_acc': ours_acc_allth, 'top_acc': top_acc_allth, 
  'random_acc': random_acc_allth, 'full_irt_acc': full_irt_acc_allth,
  'ours_var': ours_var_allth, 'top_var': top_var_allth, 
  'random_var': random_var_allth, 'full_irt_var': full_irt_var_allth,
  'welldone_dist': welldone_dist, 
  'ours_acc_head': ours_acc_head, 'AA_acc_head': AA_acc_head,
  'ours_acc_tail': ours_acc_tail, 'AA_acc_tail': AA_acc_tail
}


# 結果データの保存
'''
filename = "result/result_{0:%Y%m%d_%H%M%S}.pickle".format(now)
with open(filename, 'wb') as f:
    pickle.dump(result, f)
'''

for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

'''
plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.5, color='forestgreen')
# plt.show()
'''

# タスクのあるワーカ人数をヒストグラムで
# iteration間の平均を求める
num_worker = [[], []]
for th in [0.5, 0.6, 0.7, 0.8]:
  num_worker[0].append(worker_with_task['ours'][th] / iteration_time)
  num_worker[1].append(worker_with_task['AA'][th] / iteration_time)
w = 0.4
y_ours = num_worker[0]
y_AA = num_worker[1]


x1 = [1, 2, 3, 4]
x2 = [1.3, 2.3, 3.3, 4.3]

label_x = ['0.5', '0.6', '0.7', '0.8']
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
# 1つ目の棒グラフ
plt.bar(x1, y_ours, color='blue', width=0.3, label='DI', align="center")

# 2つ目の棒グラフ
plt.bar(x2, y_AA, color='coral', width=0.3, label='AA', align="center")

# 凡例

plt.xlabel('threshold')
plt.ylabel('Number of workers with tasks')
# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
fig.legend(bbox_to_anchor=(0.15, 0.250), loc='upper left')
# plt.show()


# 推移をプロット
result_acc_dic = {
  'ours': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'full_irt': full_irt_acc,
  'ours_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'full_irt_std': full_acc_std
  }

result_var_dic = {
  'ours': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'full_irt': full_irt_var,
  'ours_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'full_irt_std': full_var_std
  
}

result_plot_1(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_1(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

# トレードオフのグラフ

ours_trade = var_acc_plot(ours_var, ours_acc)
top_trade = var_acc_plot(top_var, top_acc)
random_trade = var_acc_plot(random_var, random_acc)

plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('variance')
ax.set_ylabel('accuracy')
    
ax.plot(ours_trade[0], ours_trade[1], color='blue', label='ours')
# ax.plot(top_trade[0], top_trade[1], color='blue', label='top')
# ax.plot(random_trade[0], random_trade[1], color='green', label='random')
# 
ax.plot(threshold, full_irt, color='purple', label='IRT')
# plt.show()


