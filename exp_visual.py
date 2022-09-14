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

AA_acc_allth = []
AA_var_allth = []
AA_tp_allth = []


random_acc_allth = []
random_var_allth = []

full_irt_acc_allth = []
full_irt_var_allth = []
PI_margin_allth = []

threshold = list([i / 100 for i in range(50, 71)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)
ours_output_alliter = {}
full_output_alliter = {}

top_assignment_allth = {}
for th in threshold:
  top_assignment_allth[th] = []
  
# 承認タスクとテストタスクを分離
qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
  qualify_dic[qt] = list(input_df.T[qt])

q_data = np.array(list(qualify_dic.values()))
params = run_girth_rasch(q_data, task_list, worker_list)
full_item_param = params[0]
full_user_param = params[1]

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 20
worker_with_task = {'ours': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}, 'AA': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}}
for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []
  ours_tp_perth = []
  DI_margin_perth = []

  top_acc_perth = []
  top_var_perth = []

  AA_acc_perth = []
  AA_var_perth = []
  AA_tp_perth = []

  random_acc_perth = []
  random_var_perth = []
  full_irt_acc_perth = []
  full_irt_var_perth = []
  PI_margin_perth = []

  while True:
    sample = devide_sample(task_list, worker_list)
    qualify_task = sample['qualify_task']
    test_task = sample['test_task']
    test_worker = sample['test_worker']
    if assignable_check(threshold, input_df, full_item_param, full_user_param, test_worker, test_task) == True:
      break

  # 各手法でのワーカ候補作成
  ours_output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = ours_output[0]
  user_param = ours_output[5]
  top_result = ours_output[6]
  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  AA_candidate = AA_assignment(threshold, input_df, test_worker, qualify_task, test_task)

  full_output = make_candidate_all(threshold, input_df, full_item_param, full_user_param, test_worker, test_task)
  full_irt_candidate = full_output[0]
  # full_item_param = full_output[1]
  # full_user_param = full_output[2]
  # top_result = full_output[3]

  for th in top_result:
    top_assignment_allth[th].append(top_result[th])

  # 保存用
  ours_output_alliter[iteration] = ours_output
  full_output_alliter[iteration] = full_output

  # worker_c_th = {th: {task: [workers]}}
  for th in ours_candidate:
    candidate_dic = ours_candidate[th]
    
    assign_dic_opt = {}
    assigned = optim_assignment(candidate_dic, test_worker, test_task, user_param)

    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    # print('th'+str(th)+'assignment size'+str(len(assign_dic_opt)))
    # print(len(assign_dic_opt.keys()))
    if th in [0.5, 0.6, 0.7, 0.8]:
      welldone_dist[th] += welldone_count(th, assign_dic_opt, full_user_param, full_item_param) / len(test_task) 
      # 割当て候補人数を数える
      worker_with_task_list = []
      for worker_set in ours_candidate[th].values():
        for worker in worker_set:
          if worker not in worker_with_task_list:
            worker_with_task_list.append(worker)
      
      worker_with_task['ours'][th] += len(worker_with_task_list)
  
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    print("DI assignment", th, acc, len(assign_dic_opt))
  
    DI_margin_sum = 0
    for task in assign_dic_opt:
      worker = assign_dic_opt[task]
      DI_margin = full_user_param[worker] - full_item_param[task]
      DI_margin_sum += DI_margin
      # print(full_item_param[task], full_user_param[worker], full_user_param[worker] - full_item_param[task], input_df[worker][task])
    print(DI_margin_sum / len(assign_dic_opt))
    DI_margin_mean = DI_margin_sum / len(assign_dic_opt)
    print('==========================================================')

    # 割当て結果分散を求める
    var = task_variance(assign_dic_opt, test_worker)
    # 割当て結果のTPを求める
    tp = calc_tp(assign_dic_opt, test_worker)
    
    ours_acc_perth.append(acc)
    ours_var_perth.append(var)
    ours_tp_perth.append(tp)
    DI_margin_perth.append(DI_margin_mean)
  
  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)
  ours_tp_allth.append(ours_tp_perth)
  DI_margin_allth.append(DI_margin_perth)
  
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
    assigned = optim_assignment(candidate_dic, test_worker, test_task, user_param)
    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    
    # print(th, len(assign_dic_opt))
      
    if th in [0.5, 0.6, 0.7, 0.8]:
      welldone_dist[th] += welldone_count(th, assign_dic_opt, full_user_param, full_item_param) / len(test_task) 
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
    tp = calc_tp(assign_dic_opt, test_worker)
  

    AA_acc_perth.append(acc)
    AA_var_perth.append(var)
    AA_tp_perth.append(tp)

  AA_acc_allth.append(AA_acc_perth)
  AA_var_allth.append(AA_var_perth)
  AA_tp_allth.append(AA_tp_perth)
  
  for th in full_irt_candidate:
    candidate_dic = full_irt_candidate[th]
    assign_dic_opt = {}
    # print(candidate_dic)
    assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)
  
    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    # print(th, len(assign_dic_opt))
    
    # 割当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    print("PI assignment", th, acc, len(assign_dic_opt))  
    PI_margin_sum = 0
    for task in assign_dic_opt:
      worker = assign_dic_opt[task]
      PI_margin = full_user_param[worker] - full_item_param[task]
      PI_margin_sum += PI_margin
      # print(full_item_param[task], full_user_param[worker], full_user_param[worker] - full_item_param[task], input_df[worker][task])
    print(PI_margin_sum / len(assign_dic_opt))
    PI_margin_mean = PI_margin_sum / len(assign_dic_opt)
    print('==========================================================')
    # print(assign_dic_opt)
    var = task_variance(assign_dic_opt, test_worker)
    
    full_irt_acc_perth.append(acc)
    full_irt_var_perth.append(var)
    PI_margin_perth.append(PI_margin_mean)

  full_irt_acc_allth.append(full_irt_acc_perth)
  full_irt_var_allth.append(full_irt_var_perth)
  PI_margin_allth.append(PI_margin_perth)
 
  
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
ours_tp = [0] * len(threshold)
DI_margin_result = [0] * len(threshold) 

ours_acc_std =  []
ours_var_std =  []

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)
top_acc_std = []
top_var_std = []

AA_acc = [0] * len(threshold)
AA_var = [0] * len(threshold)
AA_tp = [0] * len(threshold)
AA_acc_std = []
AA_var_std = []

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)
random_acc_std = []
random_var_std = []

full_irt_acc = [0] * len(threshold)
full_irt_var = [0] * len(threshold)
PI_margin_result = [0] * len(threshold) 
full_acc_std = []
full_var_std = []

print('accuracy of DI for all threshold: for 1 iteration')
print(ours_acc_allth)
print('accuracy of PI for all threshold: for 1 iteration')
print(full_irt_acc_allth)

for th in range(0, len(threshold)):
  ours_acc_sum = 0
  ours_var_sum = 0
  ours_tp_sum = 0
  ours_acc_num = 0
  ours_var_num = 0
  ours_tp_num = 0
  DI_margin_sum_th = 0
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
    ours_tp_sum += ours_tp_allth[i][th]
    ours_tp_num += 1
    DI_margin_sum_th += DI_margin_allth[i][th]
  # acc, var, tpの平均を計算
  
  ours_acc[th] = ours_acc_sum / ours_acc_num
  ours_var[th] = ours_var_sum / ours_var_num 
  ours_tp[th] = ours_tp_sum / ours_tp_num
  DI_margin_result[th] = DI_margin_sum_th / iteration_time
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
  AA_tp_sum = 0
  AA_tp_num = 0
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
 
      AA_tp_sum += AA_tp_allth[i][th]
      AA_tp_num += 1
    
  # print(AA_acc_allth)
  AA_acc[th] = AA_acc_sum / AA_acc_num
  AA_var[th] = AA_var_sum / AA_var_num
  AA_tp[th]  = AA_tp_sum / AA_tp_num

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
  PI_margin_sum_th = 0
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
    PI_margin_sum_th += PI_margin_allth[i][th]
 
  full_irt_acc[th] = full_irt_acc_sum / full_irt_acc_num
  full_irt_var[th] = full_irt_var_sum / full_irt_var_num
  PI_margin_result[th] = PI_margin_sum_th / iteration_time
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  full_acc_std.append(acc_std)
  full_var_std.append(var_std)

print("Result Collected!")
for th in top_assignment_allth:
  res = top_assignment_allth[th]
  print(th, len(res), np.mean(res))
print('=====================================')
print(full_irt_acc)

'''
# 割当て結果保存:
'''
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
filename = "result/result_{0:%Y%m%d_%H%M%S}.pickle".format(now)
with open(filename, 'wb') as f:
    pickle.dump(result, f)


for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time
'''

'''
plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.5, color='forestgreen')
# plt.show()


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
ours_trade = tp_acc_plot(ours_tp, ours_acc)
AA_trade = tp_acc_plot(AA_tp, AA_acc)

plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('max number of tasks')
ax1.set_ylabel('accuracy')
ax1.set_xlim(2, 10)



# margin推移をプロット
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('threshold')
ax.set_ylabel('margin')
x = np.array(threshold)

    
ax.plot(x, DI_margin_result, color='red', label='ours')
ax.plot(x, PI_margin_result, color='purple', label='IRT')
plt.show()

ax1.plot(ours_trade[0], ours_trade[1], color='red', label='ours')
# plt.show()


# plt.rcParams["font.size"] = 22
# fig = plt.figure() #親グラフと子グラフを同時に定義
# ax2 = ax1.twinx()
#ax2.set_xlabel('max number of tasks')
#ax2.set_ylabel('1 / accuracy')
ax1.plot(AA_trade[0], AA_trade[1], color='blue', label='AA')

# plt.show()
print('====================================================================')
print(len(ours_trade[0]))



# ax.plot(threshold, full_irt, color='purple', label='IRT')
# plt.show()



