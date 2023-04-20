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

top_acc_allth = []
top_var_allth = []

AA_acc_allth = []
AA_var_allth = []
AA_tp_allth = []


random_acc_allth = []
random_var_allth = []
full_irt_acc_allth = []
full_irt_var_allth = []

threshold = list([i / 100 for i in range(50, 81)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)

ours_output_alliter = {}
full_output_alliter = {}

top_assignment_allth = {}
for th in threshold:
  top_assignment_allth[th] = []
  
# B割り当てのタスク数を数える配列
imp_task_num_DI = []
imp_task_num_PI = []
# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 20

for iteration in range(0, iteration_time):

  ours_acc_perth = []
  ours_var_perth = []
  ours_tp_perth = []

  top_acc_perth = []
  top_var_perth = []

  AA_acc_perth = []
  AA_var_perth = []
  AA_tp_perth = []

  random_acc_perth = []
  random_var_perth = []
  full_irt_acc_perth = []
  full_irt_var_perth = []

  
  
  

  


  # 各手法でのワーカ候補作成
  ours_output =  make_candidate_imp(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = ours_output[0]
  user_param = ours_output[5]
  top_result = ours_output[6]
  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  AA_candidate = AA_assignment(threshold, input_df, test_worker, qualify_task, test_task)

  full_output = make_candidate_all_imp(threshold, input_df, task_list, worker_list, test_worker, test_task)
  full_irt_candidate = full_output[0]
  full_item_param = full_output[1]
  full_user_param = full_output[2]
  # top_result = full_output[3]
  

  for th in top_result:
    top_assignment_allth[th].append(top_result[th])

  # 保存用
  ours_output_alliter[iteration] = ours_output
  full_output_alliter[iteration] = full_output
  
  imp_task_num_DI_iter = []
  
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
    imp_task_num_DI_iter.append(len(assign_dic_opt))
    

  
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    # 割当て結果分散を求める
    var = task_variance(assign_dic_opt, test_worker)
    # 割当て結果のTPを求める
    tp = calc_tp(assign_dic_opt, test_worker)
    
    ours_acc_perth.append(acc)
    ours_var_perth.append(var)
    ours_tp_perth.append(tp)
  
  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)
  ours_tp_allth.append(ours_tp_perth)
  imp_task_num_DI.append(imp_task_num_DI_iter)
  
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
  
  imp_task_num_PI_iter = []
  for th in full_irt_candidate:
    candidate_dic = full_irt_candidate[th]
    assign_dic_opt = {}

    assigned = optim_assignment(candidate_dic, test_worker, test_task, full_user_param)
  
    for worker in assigned:
      for task in assigned[worker]:
        assign_dic_opt[task] = worker
    
    imp_task_num_PI_iter.append(len(assign_dic_opt))
    # print(th, len(assign_dic_opt))
 
    # 割当て結果の精度を求める
    acc = accuracy(assign_dic_opt, input_df)
    # print("full-irt assignment")  
    # print(assign_dic_opt)
    var = task_variance(assign_dic_opt, test_worker)
    
    full_irt_acc_perth.append(acc)
    full_irt_var_perth.append(var)

  full_irt_acc_allth.append(full_irt_acc_perth)
  full_irt_var_allth.append(full_irt_var_perth)
  imp_task_num_PI.append(imp_task_num_PI_iter)
 
  
  for th in range(0, len(threshold)):
    assign_dic = random_assignment(test_task, test_worker)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    random_acc_perth.append(acc)
    random_var_perth.append(var)

  random_acc_allth.append(random_acc_perth)
  random_var_allth.append(random_var_perth)



print('accuracy of DI for all threshold: for 1 iteration')
print(ours_acc_allth)
print(imp_task_num_DI)
print('accuracy of PI for all threshold: for 1 iteration')
print(full_irt_acc_allth)
print(imp_task_num_PI)