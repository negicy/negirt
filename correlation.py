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

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 1
worker_with_task = {'ours': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}, 'AA': {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}}
# 各ワーカーの全体正解率
skill_rate_dic = {}
# 各ワーカーの全体正解率
skill_rate_dic = {}
for w in worker_list:
  correct = 0
  task_num = 0
  for i in task_list:
    task_num += 1
    if input_df[w][i] == 1:
      correct += 1
  skill_rate_dic[w] = (correct / task_num)


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
  
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']

  
  full_output = make_candidate_all(threshold, input_df, task_list, worker_list, test_worker, test_task)
  full_irt_candidate = full_output[0]
  full_item_param = full_output[1]
  full_user_param = full_output[2]

  rate_list = []
  theta_list = []

  for worker in skill_rate_dic:
    rate_list.append(skill_rate_dic[worker])
    theta_list.append(full_user_param[worker])
  
  plt.rcParams["font.size"] = 22
  fig = plt.figure() #親グラフと子グラフを同時に定義
  #ax1 = fig.add_subplot()
  #ax1.set_xlabel('max number of tasks')
  #ax1.set_ylabel('1 / accuracy')
  #ax1.set_xlim(0, 15)

  #ax1.plot(rate_list, theta_list, color='blue', label='ours')
  plt.scatter(theta_list, rate_list, color='blue', label='ours')
  plt.show()
  print(len(rate_list))
  print(len(theta_list))
  print(skill_rate_dic)


  