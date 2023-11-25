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
import scipy.stats as stats
#from scrap.survey import *

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
# 各タスクのーの全体正解率
item_rate_dic = {}

for i in task_list:
  correct = 0
  worker_num = 0
  for w in worker_list:
    worker_num += 1
    if input_df[w][i] == 1:
      correct += 1
  item_rate_dic[i] = (correct / worker_num)

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

  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']
  
  # 承認タスクとテストタスクを分離
  qualify_task = task_list

  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  #print(q_data)
 
  # t_worker = worker_list
  # q_data = np.array(list(qualify_dic.values()))

  # params = run_girth_rasch(q_data, task_list, worker_list)
  # twoplによる推定
  params_twopl = run_girth_twopl(q_data, task_list, worker_list)
  params_onepl = run_girth_onepl(q_data, task_list, worker_list)  

  item_param_twopl = params_twopl[0]
  #print(params_twopl)
  discrimination = params_twopl[2]
  item_param_onepl = params_onepl[0]

  PI_twopl = {}
  PI_onepl = {}
  item_rate_list = []
  beta_list_twopl = []
  beta_list_onepl = []

  for item in item_rate_dic:
    beta_list_twopl.append(item_param_twopl[item])
    beta_list_onepl.append(item_param_onepl)
    item_rate_list.append(item_rate_dic[item])
    PI_twopl[item] = item_param_twopl[item]
    PI_onepl[item] = item_param_onepl[item]

  user_param_twopl = params_twopl[1]
  user_param_onepl = params_onepl[1]
  #print(params_twopl)

  parameter_dict = {'theta': user_param_twopl, 'beta': item_param_twopl, 'alpha': discrimination}

  # pickleで保存
  with open('parameter_twopl.pickle', 'wb') as f:
    pickle.dump(parameter_dict, f)

  print(discrimination)