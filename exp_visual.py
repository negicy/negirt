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
top_acc_allth = []
top_var_allth = []
random_acc_allth = []
random_var_allth = []
full_irt_acc_allth = []
full_irt_var_allth = []

threshold = list([i / 100 for i in range(50, 81)])

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 3
for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []
  top_acc_perth = []
  top_var_perth = []
  random_acc_perth = []
  random_var_perth = []
  full_irt_acc_perth = []
  full_irt_var_perth = []
  
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']

  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]

  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  full_irt_candidate = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)[0]

  # worker_c_th = {th: {task: [workers]}}
  for candidate_dic in ours_candidate.values():
    # print(candidate_dic)
    assign_dic = assignment(candidate_dic, test_worker)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    ours_acc_perth.append(acc)
    ours_var_perth.append(var)

  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)

  
  for candidate_dic in top_candidate.values():
    # print(candidate_dic)
    index = 0
    assign_dic = {}
    for task in candidate_dic:
      index = index%5
      candidate_list = candidate_dic[task]
      assign_dic[task] = random.choice(candidate_list)
      index += 1

    #assign_dic = assignment(candidate_dic, test_worker)
   
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    top_acc_perth.append(acc)
    top_var_perth.append(var)

  top_acc_allth.append(top_acc_perth)
  top_var_allth.append(top_var_perth)

  for candidate_dic in full_irt_candidate.values():
    assign_dic = assignment(candidate_dic, test_worker)
    # 割当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
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
  random_acc_sum = 0
  random_var_sum = 0
  random_acc_num = 0
  random_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
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


# 推移をプロット
result_acc_dic = {
  'ours': ours_acc, 'top': top_acc, 'random': random_acc, 'full_irt': full_irt_acc,
  'ours_std': ours_acc_std, 'top_std': top_acc_std, 'random_std': random_acc_std, 'full_irt_std': full_acc_std
  }

result_var_dic = {
  'ours': ours_var, 'top': top_var, 'random': random_var, 'full_irt': full_irt_var,
  'ours_std': ours_var_std, 'top_std': top_var_std, 'random_std': random_var_std, 'full_irt_std': full_var_std
}

result_plot(threshold, result_acc_dic, ay='accuracy').show()
result_plot(threshold, result_var_dic, ay='variance').show()

