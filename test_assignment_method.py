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
import datetime
now = datetime.datetime.now()
import csv

path = os.getcwd()


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

qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
  qualify_dic[qt] = list(input_df.T[qt])

q_data = np.array(list(qualify_dic.values()))
params = run_girth_rasch(q_data, task_list, worker_list)
full_item_param = params[0]
full_user_param = params[1]

test_assign_dic = {}

test_assign_dic[task_list[30]] = worker_list[30]
test_assign_dic[task_list[21]] = worker_list[21]
test_assign_dic[task_list[42]] = worker_list[42]
test_assign_dic[task_list[15]] = worker_list[15]
test_task = list(test_assign_dic.keys())
test_worker = list(test_assign_dic.values())

# 割り当てアルゴリズムテスト
# ワーカ候補リスト作成

worker_c = {
    task_list[30]: [worker_list[30], worker_list[21], worker_list[42], worker_list[15]],
    task_list[21]: [worker_list[30], worker_list[21], worker_list[42]],
    task_list[42]: [worker_list[30], worker_list[21]],
    task_list[15]: [worker_list[30]]
}

test_assigned = optim_assignment(worker_c, test_worker, test_task, full_user_param)

for worker in test_assigned:
      for task in test_assigned[worker]:
        test_assign_dic[task] = worker

# 割り当て結果計算: テストケース作成

print(test_assign_dic)
# 回答のか確認
for task in test_assign_dic.keys():
    worker = test_assign_dic[task]
    print(input_df[worker][task])
    print(full_user_param[worker], full_item_param[task])

# 正解率の計算
acc = accuracy(test_assign_dic, input_df)
var = task_variance(test_assign_dic, test_worker)
tp = calc_tp(test_assign_dic, worker_list)
print(acc)
print(var)
print(tp)

# welldoneカウント

wd = welldone_count(0.5, test_assign_dic, full_user_param, full_item_param)
print(wd)

