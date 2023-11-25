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
  #params_twopl = run_girth_twopl(q_data, task_list, worker_list)
  params_onepl = run_girth_onepl(q_data, task_list, worker_list)  
  item_param_onepl = params_onepl[0]

  PI_twopl = {}
  PI_onepl = {}
  item_rate_list = []
  beta_list_twopl = []
  beta_list_onepl = []

  for item in item_rate_dic:
    beta_list_onepl.append(item_param_onepl)
    item_rate_list.append(item_rate_dic[item])
    PI_onepl[item] = item_param_onepl[item]

  user_param_onepl = params_onepl[1]
 
  skill_rate_list = []

  for worker in skill_rate_dic:
    skill_rate_list.append(skill_rate_dic[worker])

e_dict = {}
z_dict = {}
p_dict = {}
# 残差を計算
diff_list_onepl = []
diff_list_twopl = []
for worker in worker_list:
  diff_onepl_list = []
  diff_twopl_list = []
  e_dict[worker] = {}
  z_dict[worker] = {}
  p_dict[worker] = {}
  for task in task_list:
    
    p_onepl = OnePLM(item_param_onepl[task], user_param_onepl[worker])
    diff_onepl = input_df[worker][task] - p_onepl
    diff_onepl_list.append(diff_onepl)

    e_ij = diff_onepl
    z_ij = e_ij / np.sqrt(p_onepl*(1 - p_onepl))

    p_dict[worker][task] = p_onepl
    e_dict[worker][task] = e_ij
    z_dict[worker][task] = z_ij

  diff_onepl_mean = np.mean(diff_onepl_list)
  diff_list_onepl.append(diff_onepl_mean)

outfit_dict = {}
infit_dict = {}
#print(z_dict)
for i in worker_list:
  outfit_dict[i] = 0
  outfit_list = []
  for j in task_list:
    outfit_list.append(z_dict[i][j]**2)
  outfit_dict[i] = np.mean(outfit_list)

eij2_sum = {}
pij_sum = {}
infit_dict = {}

for i in worker_list:
  eij2_sum[i] = 0
  pij_sum[i] = 0
  infit_dict[i] = 0
  for j in task_list:
    eij2_sum[i] += e_dict[i][j]**2
    pij_sum[i] += p_dict[i][j]*(1 - p_dict[i][j])

  infit_dict[i] = eij2_sum[i] / pij_sum[i]

under_fit_count = 0
under_fit_list = []
for i in worker_list:
  if infit_dict[i] > 1.3 or outfit_dict[i] > 1.3:
    under_fit_count += 1
    under_fit_list.append(i)
    print("=================")
    print(f'under fit:{j}')
    print(f'infit:{infit_dict[i]}')
    print(f'outfit:{outfit_dict[i]}')


print(under_fit_count)
# outfit_dictを降順にソート
outfit_dict = dict(sorted(outfit_dict.items(), key=lambda x:x[1], reverse=True))
infit_dict = dict(sorted(infit_dict.items(), key=lambda x:x[1], reverse=True))

infit_worker_list_sorted = list(infit_dict.keys())
outfit_worker_list_sorted = list(outfit_dict.keys())
# outfit_dictの上位10件を取得
print(outfit_worker_list_sorted[:10])
print(infit_worker_list_sorted[:10])

# 削除対象項目リスト
delete_worker_list = []
for worker in infit_worker_list_sorted[:20]:
  delete_worker_list.append(worker)

for worker in outfit_worker_list_sorted[:20]:
  delete_worker_list.append(worker)

# 重複する要素を削除: delete_item_list
delete_worker_list = list(set(delete_worker_list))

output_dict = {'infit': infit_dict, 'outfit': outfit_dict}

# pickleに保存
with open('delete_worker_list.pickle', 'wb') as f:
  pickle.dump(output_dict, f)