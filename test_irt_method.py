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

path = os.getcwd()
'''
Real DATA
margin=0

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

spam_worker_list = [
 
  'AK9U0LQROU5LW',
  'A2CN9J57643499'
]
for worker in spam_worker_list:
  worker_list.remove(worker)


print(len(worker_list))
# 正解確率P
print(OnePLM(2, 1))
