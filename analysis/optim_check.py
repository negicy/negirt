# 割当て結果が最適かどうかを調べる
import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import girth
import random
import scipy
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

threshold = list([i / 100 for i in range(50, 81)])

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 1
for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []
  
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']
  # print(test_worker)
  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]
  user_param = output[5]
  
  
  candidate_dic = ours_candidate[0.80]
  print(candidate_dic)
  assigned = optim_assignment(candidate_dic, test_worker, test_task, user_param)
  # print(assigned)
  assign_dic = {}
  # print(candidate_dic)

  print(assigned)
  for worker in assigned.keys():
    for task in assigned[worker]:
      print(worker)
      print(task)
      assign_dic[task] = worker
  print(assign_dic)






