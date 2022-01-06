import sys, os
from assignment_method import *
from irt_method import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import girth
# from survey import *

label_df = pd.read_csv("label_df.csv", sep = ",")
input_df = pd.read_csv("input.csv", sep = ",")
# label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
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
# ワーカーリスト作成~割り当て　実行

# すべてのタスクの平均正解率
correct_rate_dic = {}
for i in task_list:
  correct = 0
  worker_num = 0
  for w in worker_list:
    worker_num += 1
    if input_df[w][i] == 1:
      correct += 1
  correct_rate_dic[i] = (correct / worker_num)
print(correct_rate_dic)

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

print(skill_rate_dic)

acc_all_th = []
var_all_th = []

df = pd.read_csv('input.csv')
df = df.set_index('qid')
# print(df)
#task_list = list(df.index)
#worker_list = list(df.columns)
threshold = list([i / 100 for i in range(50, 81)])
# data: 0/1 のndarray (2d)  
data = df.values

# Solve for parameters
iteration_time = 5
for iteration in range(0, iteration_time):
  
  acc_per_th = []
  var_per_th = []

  results = make_candidate(threshold, input_df, label_df, worker_list, task_list)
  # results = estimate_candidate(threshold, input_df, label_df, worker_list, task_list)
  # print(results)
  # results : return worker_c_th, t_worker, test_task, random_quality, random_variance, top_worker_quality, top_worker_variance
  worker_c_th = results[0]
  test_worker_set = results[1]

  # worker_c_th = {th: {task: [workers]}}
  for candidate_dic in worker_c_th.values():
    # print(candidate_dic)
    assign_dic = assignment(candidate_dic, test_worker_set)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    v = task_variance(assign_dic, test_worker_set)
  
    acc_per_th.append(acc)
    var_per_th.append(v)

  acc_all_th.append(acc_per_th)
  var_all_th.append(var_per_th)

mean_acc = [0] * len(threshold)
mean_var = [0] * len(threshold)
for acc_list in acc_all_th:
  # acc_sum = 0
  for i in range(0, len(threshold)):
    mean_acc[i] += acc_list[i]

for var_list in var_all_th:
  # var_sum = 0
  for i in range(0, len(threshold)):
    mean_var[i] += var_list[i]


for i in range(0, len(threshold)):
  mean_acc[i] = mean_acc[i] / iteration_time
  mean_var[i] = mean_var[i] / iteration_time
print(mean_acc)
print(mean_var)

# 推移をプロット
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('threshold')
#ax1 = fig.add_subplot(2, 2, 1)
ax2 = ax1.twinx()
clrs = ['b', 'orange'] 
x = np.array(threshold)
acc_height = np.array(mean_acc)
var_height = np.array(mean_var)
ax1.plot(x, var_height, color='blue')
ax2.plot(x, acc_height, color='red')
#ax.plot(left, var_height)
# plt.savefig("irt-all-e1.png")
plt.show()

