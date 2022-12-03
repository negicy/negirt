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

with open('input_data.pickle', 'rb') as f:
  input_data = pickle.load(f)
  input_df = input_data['input_df']
  worker_list = input_data['worker_list']
  task_list = input_data['task_list']


ours_acc_allth = []
ours_var_allth = [] 
ours_tp_allth = []
DI_margin_allth = []

top_acc_allth = []
top_var_allth = []
top_tp_allth = []

AA_acc_allth = []
AA_var_allth = []
AA_tp_allth = []

random_acc_allth = []
random_var_allth = []
random_tp_allth = []

PI_acc_allth = []
PI_var_allth = []
PI_tp_allth = []
PI_margin_allth = []

PI_noise1_acc_allth = []
PI_noise1_var_allth = []
PI_noise1_tp_allth = []


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

sorted_item_param = dict(sorted(full_item_param.items(), key=lambda x: x[1], reverse=True))
print(sorted_item_param)