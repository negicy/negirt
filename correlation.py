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

with open('input_data_no_spam.pickle', 'rb') as f:
  input_data = pickle.load(f)
  input_df = input_data['input_df']
  worker_list = input_data['worker_list']
  task_list = input_data['task_list']

spam_list = [
  #'ALSF1M6V28URB',
  #'A303MN1VOKQG5I',
  #'AK9U0LQROU5LW',
  #'A3S2AUQWI7XWT4'
]
for spam in spam_list:
  worker_list.remove(spam)
print(len(worker_list))


threshold = list([i / 100 for i in range(50, 81)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)

ours_output_alliter = {}
full_output_alliter = {}

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 1
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

  sample = devide_sample(task_list, worker_list, label_df)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']
  
  # full_output = make_candidate_all(threshold, input_df, task_list, worker_list, test_worker, test_task)
  # full_irt_candidate = full_output[0]
  
  # 承認タスクとテストタスクを分離
  qualify_task = task_list

  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
 

  # t_worker = worker_list
  # q_data = np.array(list(qualify_dic.values()))

  params = run_girth_rasch(q_data, task_list, worker_list)

  item_param = params[0]
  full_item_param = {}
  for item in item_param:
    full_item_param[item] = item_param[item]

  full_user_param = params[1]
  print(full_item_param)
  rate_list = []
  theta_list = []

  for worker in skill_rate_dic:
    rate_list.append(skill_rate_dic[worker])
    theta_list.append(full_user_param[worker])
  
  plt.rcParams["font.size"] = 22
  fig = plt.figure() #親グラフと子グラフを同時に定義
  plt.scatter(theta_list, rate_list, color='blue', label='ours')
  plt.show()

  sorted_skill_rate = dict(sorted(skill_rate_dic.items(), key=lambda x: x[1], reverse=True))
  sorted_item_param = dict(sorted(full_item_param.items(), key=lambda x: x[1], reverse=True))
  sorted_user_param = dict(sorted(full_user_param.items(), key=lambda x: x[1], reverse=True))

  print(sorted_skill_rate)
  print(sorted_item_param)
  print(sorted_user_param)


bins=np.linspace(-3, 3, 30)
x = list(sorted_user_param.values())
y = list(sorted_item_param.values())
# fig3 = plt.figure()
plt.hist([x, y], bins, label=['worker', 'task'])
plt.legend(loc='upper left')
plt.xlabel("IRT parameter ")
plt.ylabel("Number of tasks and workers")
plt.show()

  