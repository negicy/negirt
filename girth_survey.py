import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from girth import twopl_mml, onepl_mml, ability_mle
from assignment_method import *
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


label_df = pd.read_csv("label_df.csv", sep = ",")
input_df = pd.read_csv("input.csv", sep = ",")
#label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
# label_df = label_df.set_index('id')

batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

# origin_id: max(prob_dic)のdict作成
task_dic = {}
for k in range(0, 100):
  task_id = "q"+str(k+1)
  # 元のテストデータのIDを取り出す
  origin_index = 'Input.'+task_id
  origin_id = batch_df[origin_index][0]
  task_dic["q"+str(k+1)] = origin_id

print(task_dic)
input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)
print(q_list)

# Task IDリストの作成
task_list = list(task_dic.values())
print(task_list)

# input_dfのインデックスを置き換え
for q in q_list:
  input_df['task_id'][q] = task_dic[q]
input_df = input_df.set_index('task_id')

worker_list = list(input_df.columns)
# df = pd.read_csv('input.csv')
# df = df.set_index('qid')
# print(df)
#task_list = list(df.index)
#worker_list = list(df.columns)


data = df.values
print(data)
# Solve for parameters

estimates = onepl_mml(data)

# Unpack estimates
discrimination_estimates = estimates['Discrimination']
difficulty_estimates = estimates['Difficulty']


#print(discrimination_estimates)
# print(difficulty_estimates)

a_list = []
for i in range(100):
    a_list.append(discrimination_estimates)

abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
# print(abilitiy_estimates)

user_param = {}
item_param = {}

for k in range(100):
    task_id = task_list[k]
    worker_id = worker_list[k]
    item_param[task_id] = difficulty_estimates[k]
    user_param[worker_id] = abilitiy_estimates[k] 

print(item_param)
print(user_param)

# ワーカーの能力分布調べる
# output: worker_id: theta の辞書 => user_param
# visual: 


# 各タスクの正答率求める

# タスクごとの各変数
avg_task_score = {}

for i in item_param:
  score = 0
  prob_sum = 0
  theta_sum = 0
  u_num = 0
  avg_task_score[i] = []
  a = a_list[0]
  b = item_param[i]
  
  avg_task_score[i].append(a)
  avg_task_score[i].append(b)
  for u in user_param:
    theta = user_param[u]
    theta_sum += theta

    u_num += 1
    if input_df[u][i] == 1:
      score += 1
    prob_sum += TwoPLM(a, b, theta)
  #avg_theta = theta_sum / u_num
  # avg_task_score[i].append(avg_theta)
  avg_task_score[i].append(score/u_num)
  avg_task_score[i].append(prob_sum/u_num)

print(avg_task_score)
x = []
y = []
for l in avg_task_score.values():
  print(l)
  x.append(l[1])
  y.append(l[2])
print(x)
print(y)

# タスクの難易度と平均正解率の散布図

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y)

