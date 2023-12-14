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

  # params = run_girth_rasch(q_data, task_list, worker_list)
  # twoplによる推定
  params_twopl = run_girth_twopl(q_data, task_list, worker_list)
  params_onepl = run_girth_onepl(q_data, task_list, worker_list)  

  item_param_twopl = params_twopl[0]
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
 
  skill_rate_list = []
  theta_list_twopl = []

  for worker in skill_rate_dic:
    skill_rate_list.append(skill_rate_dic[worker])
    theta_list_twopl.append(user_param_twopl[worker])
  
  plt.rcParams["font.size"] = 22
  fig = plt.figure() #親グラフと子グラフを同時に定義
  plt.scatter(beta_list_twopl, item_rate_list, color='blue', label='ours')
  plt.show()

  fig = plt.figure() #親グラフと子グラフを同時に定義
  plt.scatter(theta_list_twopl, skill_rate_list, color='blue', label='ours')
  plt.show()
  print(skill_rate_dic)
  print(params_twopl)
  sorted_skill_rate = dict(sorted(skill_rate_dic.items(), key=lambda x: x[1], reverse=True))
  sorted_item_param = dict(sorted(params_twopl[0].items(), key=lambda x: x[1], reverse=True))
  sorted_user_param = dict(sorted(user_param_twopl.items(), key=lambda x: x[1], reverse=True))

  print(sorted_skill_rate)
  print(sorted_item_param)
  print(sorted_user_param)


bins=np.linspace(-7, 7, 30)
x = list(sorted_user_param.values())
y = list(sorted_item_param.values())
# fig3 = plt.figure()
plt.hist([x, y], bins, label=['worker', 'task'])
plt.legend(loc='upper left')
plt.xlabel("IRT parameter ")
plt.ylabel("Number of tasks and workers")
plt.show()

'''
e_dict = {}
z_dict = {}
# 残差を計算
diff_list_onepl = []
diff_list_twopl = []
for task in task_list:
  diff_onepl_list = []
  diff_twopl_list = []
  for worker in worker_list:
    p_onepl = OnePLM(item_param_onepl[task], user_param_onepl[worker])
    p_twopl = OnePLM(item_param_twopl[task], user_param_twopl[worker])
    diff_onepl = input_df[worker][task] - p_onepl
    diff_twopl = input_df[worker][task] - p_twopl
    diff_onepl_list.append(diff_onepl)
    diff_twopl_list.append(diff_twopl)

    e_ij = diff_onepl
    z_ij = e_ij / np.sqrt(p_onepl*(1 - p_onepl))

    e_dict[worker][task] = e_ij
    z_dict[worker][task] = z_ij


  diff_onepl_mean = np.mean(diff_onepl_list)
  diff_twopl_mean = np.mean(diff_twopl_list)

  diff_list_onepl.append(diff_onepl_mean)
  diff_list_twopl.append(diff_twopl_mean)

outfit_dict = {}
infit_dict = {}

for task in task_list:
  outfit_list = []
  for worker in worker_list:
    outfit_list.append(z_dict[worker][task]**2)
  outfit_dict[task] = np.mean(outfit_list)

eij2_sum = {}
pij_sum = {}
infit_dict = {}
for task in task_list:
  eij2_sum[task] = 0
  p_ij_sum[task] = 0
  for worker in worker_list:
    eij2_sum[task] += e_dict[worker][task]**2
    pij_sum[task] += p_dict[worker][task]*(1 - p_dict[worker][task])

  infit_dict[task] = eij2_sum[task] / pij_sum[task]

for task in task_list:
  if infit_dict[task] > 1.3 or outfit[task] > 1.3:
    print(f'under fit:{task}')
    print(f'infit:{infit_dict[task]}')
    print(f'outfit:{outfit_dict[task]}')


x = diff_list_onepl
y = diff_list_twopl
# fig3 = plt.figure()
bins=np.linspace(-1, 1, 30)
plt.hist([x, y], bins, label=['onePLM', 'twoPLM'])
plt.legend(loc='upper left')
plt.xlabel("difference between true and estimated parameter")
plt.ylabel("Number of assignments")
#plt.show()

print(np.mean(diff_list_onepl))
print(np.mean(diff_list_twopl))

# 残差を箱ひげ図で表示
fig = plt.figure()
plt.boxplot([diff_list_onepl, diff_list_twopl])
plt.xlabel("IRT model")
plt.ylabel("residual")
plt.xticks([1, 2], ['onePLM', 'twoPLM'])
#plt.show()
'''