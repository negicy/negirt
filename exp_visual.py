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
from survey import *

path = os.getcwd()
'''
Real DATA
'''

'''
データ準備
'''
label_df = pd.read_csv("label_df.csv", sep = ",")
batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

with open('input_data.pickle', 'rb') as f:
  input_data = pickle.load(f)
  input_df = input_data['input_df']
  worker_list = input_data['worker_list']
  task_list = input_data['task_list']


ours_acc_allth = []
ours_var_allth = [] 
ours_tp_allth = []
DI_margin_allth = []

top_acc_allth = []
top_var_allth = []
top_tp_allth = []

AA_acc_allth = []
AA_var_allth = []
AA_tp_allth = []

random_acc_allth = []
random_var_allth = []
random_tp_allth = []

PI_acc_allth = []
PI_var_allth = []
PI_tp_allth = []
PI_margin_allth = []

PI_noise1_acc_allth = []
PI_noise1_var_allth = []
PI_noise1_tp_allth = []


threshold = list([i / 100 for i in range(50, 81)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)
ours_output_alliter = {}
full_output_alliter = {}

top_assignment_allth = {}
for th in threshold:
  top_assignment_allth[th] = []
  

# 承認タスクとテストタスクを分離
# PIでのパラメータ推定
qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
  qualify_dic[qt] = list(input_df.T[qt])

q_data = np.array(list(qualify_dic.values()))
params = run_girth_rasch(q_data, task_list, worker_list)
full_item_param = params[0]
full_user_param = params[1]


'''
イテレーション
'''
# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 20
worker_with_task = {'ours': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}, 'AA': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}}
for iteration in range(0, iteration_time):
  print('============|', iteration, "|===============")
  ours_acc_perth = []
  ours_var_perth = []
  ours_tp_perth = []
  DI_margin_perth = []

  top_acc_perth = []
  top_var_perth = []
  top_tp_perth = []

  AA_acc_perth = []
  AA_var_perth = []
  AA_tp_perth = []

  random_acc_perth = []
  random_var_perth = []
  random_tp_perth = []

  PI_acc_perth = []
  PI_var_perth = []
  PI_tp_perth = []
  PI_margin_perth = []

  PI_noise1_acc_perth = []
  PI_noise1_var_perth = []
  PI_noise1_tp_perth = []
  
  
  # Assignable taskのみをサンプリング
  while True:
    sample = devide_sample(task_list, worker_list, label_df)
    qualify_task = sample['qualify_task']
    test_task = sample['test_task']
    test_worker = sample['test_worker']
    if assignable_check(threshold, input_df, full_item_param, full_user_param, test_worker, test_task) == True:
      break
    else:
      break
  
  # 各手法でのワーカ候補作成
  ours_output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = ours_output[0]
  user_param = ours_output[5]
  top_result = ours_output[6]
  DI_item_param = ours_output[4]

  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)

  AA_output = AA_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  AA_candidate = AA_output[0]
  AA_top_workers_dict = AA_output[1]

  full_output = make_candidate_all(threshold, input_df, full_item_param, full_user_param, test_worker, test_task)
  PI_candidate = full_output[0]
  
  PI_noise1 = make_candidate_PI_noise(threshold, input_df, full_item_param, full_user_param, test_worker, test_task)
  PI_noise1_candidate = PI_noise1[0]

  for th in top_result:
    top_assignment_allth[th].append(top_result[th])

  # 保存用
  ours_output_alliter[iteration] = ours_output
  full_output_alliter[iteration] = full_output

  for th in PI_candidate:
    candidate_dic = PI_candidate[th]
    PI_assign_dic_opt_A = {}
    PI_assign_dic_opt_NA = {}
    PI_assign_dic_opt = {}

    PI_assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)
  
    for worker in PI_assigned:
      for task in PI_assigned[worker]:
        PI_assign_dic_opt[task] = worker
        PI_assign_dic_opt_A[task] = worker
    # print(th, len(assign_dic_opt))

    # NA タスクをランダム割当て
    # print('PI_size:', len(PI_assign_dic_opt_A))
    top_workers = sort_test_worker(test_worker, full_user_param, N=10)
    count = 0
    for task in test_task:
      if task not in PI_assign_dic_opt.keys():
        assigned_worker = random.choice(top_workers)
        PI_assign_dic_opt[task] = assigned_worker
        PI_assign_dic_opt_NA[task] = assigned_worker
    
    #print('DI-PIのNA:', accuracy(DI_assign_dic_PI_NA, input_df), len(DI_assign_dic_PI_NA), th)
    # DI_assign_dic_opt = DI_assign_dic_PI_NA
    # PI_assign_dic_opt = PI_assign_dic_opt_NA

    #print('DI-NA:', accuracy(DI_assign_dic_opt_NA, input_df), len(DI_assign_dic_opt_NA), th)
    #print('DI-A:', accuracy(DI_assign_dic_opt_A, input_df), len(DI_assign_dic_opt_A), th)
    # 割当て結果の精度を求める
    
    acc = accuracy(PI_assign_dic_opt, input_df)
    # acc = accuracy(PI_assign_dic_opt_A, input_df)
   
    PI_margin_sum = 0
    for task in PI_assign_dic_opt_NA:
      worker = PI_assign_dic_opt_NA[task]
     
      PI_margin = full_user_param[worker] - full_item_param[task]
      PI_margin_sum += PI_margin

    PI_margin_mean = PI_margin_sum / len(PI_assign_dic_opt)
 
    var = task_variance(PI_assign_dic_opt, test_worker)
    tp = calc_tp(PI_assign_dic_opt, test_worker)
    
    PI_acc_perth.append(acc)
    PI_var_perth.append(var)
    PI_tp_perth.append(tp)
    PI_margin_perth.append(PI_margin_mean)

  PI_acc_allth.append(PI_acc_perth)
  PI_var_allth.append(PI_var_perth)
  PI_tp_allth.append(PI_tp_perth)
  PI_margin_allth.append(PI_margin_perth)

  # worker_c_th = {th: {task: [workers]}}
  for th in ours_candidate:
    candidate_dic = ours_candidate[th]
    
    DI_assign_dic_opt_NA = {}
    DI_assign_dic_opt_A = {}
    DI_assign_dic_opt = {}
    DI_assigned = optim_assignment(candidate_dic, test_worker, test_task, user_param)

    for worker in DI_assigned:
      for task in DI_assigned[worker]:
        DI_assign_dic_opt[task] = worker
        DI_assign_dic_opt_A[task] = worker
 
    # print('th'+str(th)+'assignment size'+str(len(assign_dic_opt)))
    ## NAが見つけたAタスクのaccuracyを調べる
    # NA タスクをランダム割当て
    DI_NA_count = 0
    top_workers = sort_test_worker(test_worker, user_param, N=10)
    # print('DI_size:', len(DI_assign_dic_opt))
    for task in test_task:
      if task not in DI_assign_dic_opt.keys():
        DI_NA_count += 1
        assigned_worker = random.choice(top_workers)
        DI_assign_dic_opt[task] = assigned_worker
        DI_assign_dic_opt_NA[task] = assigned_worker
    ## DIが見つけたNAタスクの正解率を調べる．
    
    # print(DI_NA_count, len(qualify_task), len(DI_assign_dic_opt))
    
    DI_assign_dic_PI_NA = {}
    DI_assign_dic_PI_A = {}
    for task in test_task:
      if task in PI_assign_dic_opt_NA:
        DI_assign_dic_PI_NA[task] = DI_assign_dic_opt[task]
      if task in PI_assign_dic_opt_A:
        DI_assign_dic_PI_A[task] = DI_assign_dic_opt[task]

    # print('DI size-2:', len(DI_assign_dic_opt), th)
    # print('PI size:', len(PI_assign_dic_opt), th)
    #print('PI-NA:', accuracy(PI_assign_dic_opt_NA, input_df), len(PI_assign_dic_opt_NA), th)
    # print('PI-A:', accuracy(PI_assign_dic_opt_A, input_df), len(PI_assign_dic_opt_A), th)
    #print('DI-PI-NA', accuracy(DI_assign_dic_PI_NA, input_df), len(DI_assign_dic_PI_NA))


    # 割り当て結果の精度を求める
    # acc = accuracy(DI_assign_dic_PI_NA, input_df)
    acc = accuracy(DI_assign_dic_opt, input_df)
    # print("DI assignment", th, acc, len(assign_dic_opt))
    DI_margin_sum = 0
    
    for task in DI_assign_dic_PI_NA:
      worker = DI_assign_dic_PI_NA[task]
      DI_margin = user_param[worker] - full_item_param[task]
      print(user_param[worker])
      DI_margin_sum += DI_margin
      # print(full_item_param[task], full_user_param[worker], full_user_param[worker] - full_item_param[task], input_df[worker][task])

    # DI_mse = math.sqrt(DI_margin_sum / len(DI_assign_dic_opt))
    DI_mean = DI_margin_sum / len(DI_assign_dic_opt)
    # print(f'DI_margin_sum: {DI_margin_sum}')
    # print(DI_margin_mean)
    # 割当て結果分散を求める
    var = task_variance(DI_assign_dic_opt, test_worker)
    # 割当て結果のTPを求める
    tp = calc_tp(DI_assign_dic_opt, test_worker)

    ours_acc_perth.append(acc)
    ours_var_perth.append(var)
    ours_tp_perth.append(tp)
    DI_margin_perth.append(DI_mean)
  
  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)
  ours_tp_allth.append(ours_tp_perth)
  DI_margin_allth.append(DI_margin_perth)
 
  for candidate_dic in top_candidate.values():
    index = 0
    top_assign_dic = {}
    for task in candidate_dic:
      index = index % 5
      candidate_list = candidate_dic[task]
      top_assign_dic[task] = random.choice(candidate_list)
      index += 1
   
    # 割り当て結果の精度を求める
    acc = accuracy(top_assign_dic, input_df)
    var = task_variance(top_assign_dic, test_worker)
    tp = calc_tp(top_assign_dic, test_worker)
  
    top_acc_perth.append(acc)
    top_var_perth.append(var)
    top_tp_perth.append(tp)

  top_acc_allth.append(top_acc_perth)
  top_var_allth.append(top_var_perth)
  top_tp_allth.append(top_tp_perth)

  for th in AA_candidate:
    candidate_dic = AA_candidate[th]
    
    AA_assign_dic_opt = {}
  
    AA_assigned = optim_assignment(candidate_dic, test_worker, test_task, AA_top_workers_dict)
    for worker in AA_assigned:
      for task in AA_assigned[worker]:
        AA_assign_dic_opt[task] = worker
        
    if th in [0.5, 0.6, 0.7, 0.8]:
      welldone_dist[th] += welldone_count(th, AA_assign_dic_opt, full_user_param, full_item_param) / len(test_task) 
  
      worker_with_task_list = []
      for worker_set in AA_candidate[th].values():
        for worker in worker_set:
          if worker not in worker_with_task_list:
            worker_with_task_list.append(worker)

      worker_with_task['AA'][th] += len(worker_with_task_list)
    
    AA_top_workers_list = list(AA_top_workers_dict)
    for task in test_task:
      if task not in AA_assign_dic_opt.keys():
        AA_assign_dic_opt[task] = random.choice(AA_top_workers_list[:5])
    # print('割当てサイズ', len(AA_assign_dic_opt))
     # 割り当て結果の精度を求める
    acc = accuracy(AA_assign_dic_opt, input_df)
    var = task_variance(AA_assign_dic_opt, test_worker)
    tp = calc_tp(AA_assign_dic_opt, test_worker)

    AA_acc_perth.append(acc)
    AA_var_perth.append(var)
    AA_tp_perth.append(tp)

  AA_acc_allth.append(AA_acc_perth)
  AA_var_allth.append(AA_var_perth)
  AA_tp_allth.append(AA_tp_perth)
  
  for th in PI_noise1_candidate:
    candidate_dic = PI_noise1_candidate[th]
    PI_noise1_assign_dic_opt = {}
    # assign_dic = assignment(candidate_dic, test_worker)
    assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)

    for worker in assigned:
      for task in assigned[worker]:
        PI_noise1_assign_dic_opt[task] = worker

    top_workers = sort_test_worker(test_worker, full_user_param, N=5)
    for task in test_task:
      if task not in PI_noise1_assign_dic_opt.keys():
        PI_noise1_assign_dic_opt[task] = random.choice(top_workers)
    # assign_dic = assignment(candidate_dic, test_worker)
    # 割当て結果の精度を求める

    acc = accuracy(PI_noise1_assign_dic_opt, input_df)
    var = task_variance(PI_noise1_assign_dic_opt, test_worker)
    tp = calc_tp(PI_noise1_assign_dic_opt, test_worker)

    PI_noise1_acc_perth.append(acc)
    PI_noise1_var_perth.append(var)
    PI_noise1_tp_perth.append(tp)

  PI_noise1_acc_allth.append(PI_noise1_acc_perth)
  PI_noise1_var_allth.append(PI_noise1_var_perth)
  PI_noise1_tp_allth.append(PI_noise1_tp_perth)
  
  for th in range(0, len(threshold)):
    assign_dic = random_assignment(test_task, test_worker)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
    tp = calc_tp(assign_dic, test_worker)
  
    random_acc_perth.append(acc)
    random_var_perth.append(var)
    random_tp_perth.append(tp)

  random_acc_allth.append(random_acc_perth)
  random_var_allth.append(random_var_perth)
  random_tp_allth.append(random_tp_perth)

'''
結果の計算
'''

ours_acc = [0] * len(threshold)
ours_var = [0] * len(threshold)
ours_tp = [0] * len(threshold)
DI_margin_result = [0] * len(threshold) 
ours_acc_std =  []
ours_var_std =  []

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)
top_tp = [0] * len(threshold)
top_acc_std = []
top_var_std = []

AA_acc = [0] * len(threshold)
AA_var = [0] * len(threshold)
AA_tp = [0] * len(threshold)
AA_acc_std = []
AA_var_std = []

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)
random_tp = [0] * len(threshold)
random_acc_std = []
random_var_std = []

PI_acc = [0] * len(threshold)
PI_var = [0] * len(threshold)
PI_tp = [0] * len(threshold)
full_acc_std = []
full_var_std = []
PI_margin_result = [0] * len(threshold) 

PI_noise1_acc = [0] * len(threshold)
PI_noise1_var = [0] * len(threshold)
PI_noise1_tp = [0] * len(threshold)


ours_result = combine_iteration(threshold, iteration_time, ours_acc_allth, ours_var_allth, ours_tp_allth)
ours_acc = ours_result[0]
ours_var = ours_result[1]
ours_tp = ours_result[2]
# 標準偏差を計算
ours_acc_std = ours_result[3]
ours_var_std = ours_result[4]
ours_acc_head = ours_result[5]
ours_acc_tail = ours_result[6]

for th in range(0, len(threshold)):
  DI_margin_sum_th = 0
  for i in range(0, iteration_time):
    DI_margin_sum_th += DI_margin_allth[i][th]
    # print('Sum:', DI_margin_sum_th)
  # acc, var, tpの平均を計算
  DI_margin_result[th] = DI_margin_sum_th / iteration_time
# DI_margin_result[th] = DI_margin_sum_th / iteration_time
print(DI_margin_allth)



AA_result = combine_iteration(threshold, iteration_time, AA_acc_allth, AA_var_allth, AA_tp_allth)
AA_acc = AA_result[0]
AA_var = AA_result[1]
AA_tp = AA_result[2]
# 標準偏差を計算
AA_acc_std = AA_result[3]
AA_var_std = AA_result[4]
AA_acc_head = AA_result[5]
AA_acc_tail = AA_result[6]


random_result = combine_iteration(threshold, iteration_time, random_acc_allth, random_var_allth, random_tp_allth)
random_acc = random_result[0]
random_var = random_result[1]
random_tp = random_result[2]
# 標準偏差を計算
random_acc_std = random_result[3]
random_var_std = random_result[4]
random_acc_head = random_result[5]
random_acc_tail = random_result[6]

# print(random_acc)

top_result = combine_iteration(threshold, iteration_time, top_acc_allth, top_var_allth, top_tp_allth)
top_acc = top_result[0]
top_var = top_result[1]
top_tp = top_result[2]
# 標準偏差を計算
top_acc_std = top_result[3]
top_var_std = top_result[4]
top_acc_head = top_result[5]
top_acc_tail = top_result[6]

PI_result = combine_iteration(threshold, iteration_time, PI_acc_allth, PI_var_allth, PI_tp_allth)
PI_acc = PI_result[0]
PI_var = PI_result[1]
PI_tp = PI_result[2]
# 標準偏差を計算
PI_acc_std = PI_result[3]
PI_var_std = top_result[4]
for th in range(0, len(threshold)):
  PI_margin_sum_th = 0
  for i in range(0, iteration_time):
    PI_margin_sum_th += PI_margin_allth[i][th]
  # acc, var, tpの平均を計算
  PI_margin_result[th] = PI_margin_sum_th / iteration_time
# PI_margin_result[th] = PI_margin_sum_th / iteration_time

for th in range(0, len(threshold)):
  PI_noise1_acc_sum = 0
  PI_noise1_var_sum = 0
  PI_noise1_acc_num = 0
  PI_noise1_var_num = 0
  PI_noise1_tp_sum = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if PI_noise1_acc_allth[i][th] != "null":
      PI_noise1_acc_sum += PI_noise1_acc_allth[i][th]
      list_acc_th.append(random_acc_allth[i][th])
      PI_noise1_acc_num += 1
    if PI_noise1_var_allth[i][th] != "null":
      PI_noise1_var_sum += PI_noise1_var_allth[i][th]
      list_var_th.append(PI_noise1_var_allth[i][th])
      PI_noise1_var_num += 1
    PI_noise1_tp_sum += PI_noise1_tp_allth[i][th]
    
  PI_noise1_acc[th] = PI_noise1_acc_sum / PI_noise1_acc_num
  PI_noise1_var[th] = PI_noise1_var_sum / PI_noise1_var_num
  PI_noise1_tp[th] = PI_noise1_tp_sum / iteration_time



  # 標準偏差を計算

'''
可視化
'''

for th in top_assignment_allth:
  res = top_assignment_allth[th]
  # print(th, len(res), np.mean(res))


'''
# 割当て結果保存:
'''
import datetime
now = datetime.datetime.now()
result = {
  'ours_output': ours_output_alliter, 'full_output': full_output_alliter, 
  'ours_acc': ours_acc_allth, 'top_acc': top_acc_allth, 
  'random_acc': random_acc_allth, 'PI_acc': PI_acc_allth,
  'ours_var': ours_var_allth, 'top_var': top_var_allth, 
  'random_var': random_var_allth, 'PI_var': PI_var_allth,
  'welldone_dist': welldone_dist, 
  'ours_acc_head': ours_acc_head, 'AA_acc_head': AA_acc_head,
  'ours_acc_tail': ours_acc_tail, 'AA_acc_tail': AA_acc_tail
}


# 結果データの保存
filename = "result/result_{0:%Y%m%d_%H%M%S}.pickle".format(now)
with open(filename, 'wb') as f:
    pickle.dump(result, f)


for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

'''
# 割当て成功数ヒストグラム
plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
# ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.5, color='forestgreen')
# plt.show()


# タスクのあるワーカ人数をヒストグラムで
# iteration間の平均を求める
'''


num_worker = [[], []]
for th in [0.5, 0.6, 0.7, 0.8]:
  num_worker[0].append(worker_with_task['ours'][th] / iteration_time)
  num_worker[1].append(worker_with_task['AA'][th] / iteration_time)
w = 0.4
y_ours = num_worker[0]
y_AA = num_worker[1]


x1 = [1, 2, 3, 4]
x2 = [1.3, 2.3, 3.3, 4.3]

'''
# 少なくとも1つ以上のタスクを与えられたワーカのヒストグラム
label_x = ['0.5', '0.6', '0.7', '0.8']
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
# 1つ目の棒グラフ
# plt.bar(x1, y_ours, color='blue', width=0.3, label='DI', align="center")

# 2つ目の棒グラフ
# plt.bar(x2, y_AA, color='coral', width=0.3, label='AA', align="center")

# 凡例
plt.xlabel('threshold')
plt.ylabel('Number of workers with tasks')
# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
fig.legend(bbox_to_anchor=(0.15, 0.250), loc='upper left')
# plt.show()
'''
# 推移をプロット

result_acc_dic = {
  'ours': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_acc,
  'ours_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_acc_std
  }

result_var_dic = {
  'ours': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_var,
  'ours_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_var_std
}

result_plot_1(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_1(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

# トレードオフのグラフ
ours_trade = tp_acc_plot(ours_tp, ours_acc)
AA_trade = tp_acc_plot(AA_tp, AA_acc)
top_trade = tp_acc_plot(top_tp, top_acc)
random_trade = tp_acc_plot(random_tp, random_acc)
PI_trade = tp_acc_plot(PI_tp, PI_acc)
PI_noise1_trade = tp_acc_plot(PI_noise1_tp, PI_noise1_acc)

# top_trade = var_acc_plot(top_var, top_acc)
# random_trade = var_acc_plot(random_var, random_acc)

plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('Working Opportunity')
ax.set_ylabel('accuracy')
ax.set_xlim(0, 0.5)

bbox=(0.2750, 0.400)
ax.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
ax.plot(PI_noise1_trade[0], PI_noise1_trade[1], color='orange', label='IRT(PI0.5)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
plt.show()



# トレードオフのグラフ
ours_trade = var_acc_plot(ours_var, ours_acc)
AA_trade = var_acc_plot(AA_var, AA_acc)
top_trade = var_acc_plot(top_var, top_acc)
random_trade = var_acc_plot(random_var, random_acc)
PI_trade = var_acc_plot(PI_var, PI_acc)
PI_noise1_trade = var_acc_plot(PI_noise1_var, PI_noise1_acc)

plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('Working variance')
ax1.set_ylabel('accuracy')
ax1.set_xlim(0, 25)

bbox=(0.2750, 0.400)
ax1.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax1.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax1.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax1.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax1.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
ax1.plot(PI_noise1_trade[0], PI_noise1_trade[1], color='orange', label='IRT(PI0.5)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
#plt.show()


# 推移をプロット
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('threshold')
ax.set_ylabel('margin')
x = np.array(threshold)
print(DI_margin_result)
ax.plot(x, DI_margin_result, color='red', label='IRT(DI)')
ax.plot(x, PI_margin_result, color='purple', label='IRT(PI)')
plt.show()







