import sys, os
from assignment_method import *
from estimation_method import *
from irt_method import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import girth
# from survey import 

label_df = pd.read_csv("label_df.csv", sep = ",")
input_df = pd.read_csv("input.csv", sep = ",")

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

# print(task_dic)
input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)
# print(q_list)

# Task IDリストの作成
task_list = list(task_dic.values())
# print(task_list)

# input_dfのインデックスを置き換え
for q in q_list:
  input_df['task_id'][q] = task_dic[q]
input_df = input_df.set_index('task_id')

worker_list = list(input_df.columns)
# ワーカーリスト作成~割り当て　実行


diffs = compare_difficulty(input_df, label_df, worker_list, task_list)
# print(diffs)
category_dic = diffs[0]
print(type(category_dic))
# print(category_dic['Businness'])
irt_diff = diffs[1]
test_task = diffs[2]

# fig = plt.figure() 

# 各カテゴリの推定難易度
'''
businness_diff = category_dic['Businness']['mb']
economy_diff = category_dic['Economy']['mb']
tech_diff = category_dic['Technology&Science']['mb']
health_diff = category_dic['Health']['mb']

count = 0

diff_dist = {'Businness': [], 'Economy': [], 'Technology&Science': [], 'Health': []}
for item in test_task:
  est_category = label_df['estimate_label'][item]
  est_diff = category_dic[est_category]['mb']

  true_diff = irt_diff[item]
  diff_dist[est_category].append(est_diff - true_diff)
  
print(diff_dist)

# カテゴリの分布をヒストグラム
hist_businness =  Frequency_Distribution(diff_dist['Businness'], [-4, 4], 0.5)
hist_tech =  Frequency_Distribution(diff_dist['Technology&Science'], [-4, 4], 0.5)
hist_economy =  Frequency_Distribution(diff_dist['Economy'], [-4, 4], 0.5)
hist_health =  Frequency_Distribution(diff_dist['Health'], [-4, 4], 0.5)
# hist_tech = Frequency_Distribution(category_test_dic['Technology&Science']['d'], [-3, 3.5], 0.5) 
# hist_economy =  Frequency_Distribution(category_test_dic['Economy']['d'], [-3, 3.5], 0.5) 
# hist_healthy = Frequency_Distribution(category_test_dic['Health']['d'], [-3, 3.5], 0.5) 

fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#hist_businness.plot.bar(x='階級値', y='度数', label='number of tasks', xlabel='estimated difficulty - true difficulty')
#hist_tech.plot.bar(x='階級値', y='度数', label='number of tasks', xlabel='estimated difficulty - true difficulty')
#hist_economy.plot.bar(x='階級値', y='度数', label='number of tasks', xlabel='estimated difficulty - true difficulty')
hist_health.plot.bar(x='階級値', y='度数', label='number of tasks', xlabel='estimated difficulty - true difficulty')
plt.show()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_xlabel('difficulty of category')

# clrs = ['b', 'orange'] 
# ax1.plot(x, var_height, color='blue')
# ax1.plot(x, acc_height, color='red')

# ax.plot(left, var_height)
# plt.savefig("irt-all-e1.png")
# plt.show()

'''
acc_all_th = []
var_all_th = []

threshold = list([i / 100 for i in range(50, 81)])

# Solve for parameters
iteration_time = 10
for iteration in range(0, iteration_time):
  
  acc_per_th = []
  var_per_th = []

  results = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)
  # results = estimate_candidate(threshold, input_df, label_df, worker_list, task_list)


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

# ax2 = ax1.twinx()
clrs = ['b', 'orange'] 
x = np.array(threshold)
acc_height = np.array(mean_acc)
var_height = np.array(mean_var)
# ax1.plot(x, var_height, color='blue')

ax1.plot(x, acc_height, color='red')
# ax1.plot(x, x, color='orange')

# plt.ylim(0.6, 0.75)
# ax.plot(left, var_height)
# plt.savefig("irt-all-e1.png")
plt.show()


