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

'''
# Solve for parameters
iteration_time = 10
for iteration in range(0, iteration_time):
  
  acc_per_th = []
  var_per_th = []

  results = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)

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
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')
# ax1 = fig.add_subplot(2, 2, 1)
ax1.set_ylabel('accuracy')

ax2 = ax1.twinx()
ax2.set_ylabel('variance')
 
x = np.array(threshold)
acc_height = np.array(mean_acc)
var_height = np.array(mean_var)

ax1.plot(x, acc_height, color='red', label='accuracy')
ax2.plot(x, var_height, color='blue', label='variance')

#ax.plot(left, var_height)
fig.legend(bbox_to_anchor=(0.150, 0.880), loc='upper left')
# plt.savefig("irt-all-pl1.png")
plt.show()
'''

threshold = list([i / 100 for i in range(60, 81, 10)])


# ワーカー候補だけ作成する
# ワーカーリスト作成~割り当て　実行
ours_all_iter = []
baseline_all_iter = []
iteration_time = 1
for iteration in range(0, iteration_time):
  # results = just_candidate(threshold, label_df, worker_list)
  ours_candidate = make_candidate(threshold, input_df, label_df, worker_list, task_list)[0]
  baseline_candidate = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)[0]
  # print(results)

  ours_all_iter.append(ours_candidate)
  baseline_all_iter.append(baseline_candidate)
# print(ours_all_iter)
# ワーカ人数をカウントする辞書
baseline_num_dic = {}
ours_num_dic = {}

for th in threshold:
  baseline_num_dic[th] = 0
  ours_num_dic[th] = 0

for iter in range(0, iteration):
  for th in threshold:
    # thにおける各タスクのワーカ候補リスト巡回
    ours_worker_list = []
    baseline_worker_list = []
    for worker_list in ours_all_iter[iter][th].values():
      for worker in worker_list:
        if worker not in ours_worker_list:
          ours_worker_list.append(worker)
          
    for worker_list in baseline_all_iter[iter][th].values():
      print(worker_list)
      for worker in worker_list:
        if worker not in baseline_worker_list:
          baseline_worker_list.append(worker)  

    baseline_num_dic[th] += len(baseline_worker_list)
    ours_num_dic[th] += len(ours_worker_list)
    
print(baseline_worker_list)
# 割り当て数の平均を求める
for th in threshold:
  baseline_num = baseline_num_dic[th]
  baseline_avg = baseline_num / iteration_time

  ours_num = ours_num_dic[th]
  ours_avg = ours_num / iteration_time

  baseline_num_dic[th] = int(baseline_avg)
  ours_num_dic[th] = int(ours_avg)


#del matplotlib.font_manager.weight_dict['roman']
fig = plt.figure()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.figure(figsize=[6,4])
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['font.size'] = 16

location = ['0.6', '0.7', '0.8']
baseline = baseline_num_dic.values()
ours = ours_num_dic.values()
plt.xlabel("threshold")
plt.ylabel("number of workers with tasks")
plt.bar(location, baseline, width=-0.4, align='edge', label='baseline')
plt.bar(location, ours, width=0.4, align='edge', label='ours')


fig.tight_layout()
plt.legend(loc='lower right')
# plt.savefig("histgram-i5-s2.pdf", bbox_inches='tight')
plt.show()
