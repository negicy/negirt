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
'''
Real DATA
'''

'''
データ準備
'''
label_df = pd.read_csv("label_df.csv", sep = ",")
batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

with open('input_data_no_spam.pickle', 'rb') as f:
  input_data = pickle.load(f)
  input_df = input_data['input_df']
  worker_list = input_data['worker_list']
  task_list = input_data['task_list']


threshold = list([i / 100 for i in range(50, 81)])
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)
ours_output_alliter = {}
full_output_alliter = {}

top_assignment_allth = {}
for th in threshold:
  top_assignment_allth[th] = []
  
# 承認タスクとテストタスクを分離
# PIでのパラメータ推定
qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
  qualify_dic[qt] = list(input_df.T[qt])

q_data = np.array(list(qualify_dic.values()))
params = run_girth_rasch(q_data, task_list, worker_list)
full_item_param = params[0]
full_user_param = params[1]


# Assignable taskのみをサンプリング

sample = devide_sample(task_list, worker_list, label_df)
qualify_task = sample['qualify_task']
test_task = sample['test_task']
test_worker = sample['test_worker']

print('worker_list_size:', len(worker_list))
print('qualify_task size:', len(qualify_task))
print('test_task size:', len(test_task))

print('test_worker size:', len(test_worker))
    
print('重複チェック:', set(qualify_task) & set(test_task))
