import sys, os
from assignment_method import *
from irt_method import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import girth
import sys
import random
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

input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)


# Task IDリストの作成
task_list = list(task_dic.values())

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
# print(correct_rate_dic)

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
ours_acc_allth = []
ours_var_allth = []
top_acc_allth = []
top_var_allth = []
random_acc_allth = []
random_var_allth = []
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
# 割当て結果の比較(random, top, ours)
iteration_time = 10
for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []

  top_acc_perth = []
  top_var_perth = []

  random_acc_perth = []
  random_var_perth = []

  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']

  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]

  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)

  # worker_c_th = {th: {task: [workers]}}
  for candidate_dic in ours_candidate.values():
    # print(candidate_dic)
    assign_dic = assignment(candidate_dic, test_worker)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    ours_acc_perth.append(acc)
    ours_var_perth.append(var)

  ours_acc_allth.append(ours_acc_perth)
  ours_var_allth.append(ours_var_perth)

  
  for candidate_dic in top_candidate.values():
    # print(candidate_dic)
    index = 0
    assign_dic = {}
    for task in candidate_dic:
      index = index%5
      candidate_list = candidate_dic[task]
      assign_dic[task] = random.choice(candidate_list)
      index += 1


    #assign_dic = assignment(candidate_dic, test_worker)
    print('assign_dic:')
    print(assign_dic)
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    top_acc_perth.append(acc)
    top_var_perth.append(var)

  top_acc_allth.append(top_acc_perth)
  top_var_allth.append(top_var_perth)

  for th in range(0, len(threshold)):
  
    assign_dic = random_assignment(test_task, test_worker)
    
    #assign_dic = assignment(candidate_dic, test_worker)
  
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    random_acc_perth.append(acc)
    random_var_perth.append(var)

  random_acc_allth.append(random_acc_perth)
  random_var_allth.append(random_var_perth)

ours_acc = [0] * len(threshold)
ours_var = [0] * len(threshold)

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)

for th in range(0, len(threshold)):
  ours_acc_sum = 0
  ours_var_sum = 0
  ours_acc_num = 0
  ours_var_num = 0
  for i in range(0, iteration_time):
    #
    if ours_acc_allth[i][th] != "null":
      ours_acc_sum += ours_acc_allth[i][th]
      ours_acc_num += 1
    ours_var_sum += ours_var_allth[i][th]
    ours_var_num += 1
  ours_acc[th] = ours_acc_sum / ours_acc_num
  ours_var[th] = ours_var_sum / ours_var_num 

for th in range(0, len(threshold)):
  top_acc_sum = 0
  top_var_sum = 0
  top_acc_num = 0
  top_var_num = 0
  for i in range(0, iteration_time):
    #
    if top_acc_allth[i][th] != "null":
      top_acc_sum += top_acc_allth[i][th]
      top_acc_num += 1
    if top_var_allth[i][th] != "null":
      top_var_sum += top_var_allth[i][th]
      top_var_num += 1
    
  top_acc[th] = top_acc_sum / top_acc_num
  top_var[th] = top_var_sum / top_var_num 
  
for th in range(0, len(threshold)):
  random_acc_sum = 0
  random_var_sum = 0
  random_acc_num = 0
  random_var_num = 0
  for i in range(0, iteration_time):
    #
    if random_acc_allth[i][th] != "null":
      random_acc_sum += random_acc_allth[i][th]
      random_acc_num += 1
    if random_var_allth[i][th] != "null":
      random_var_sum += random_var_allth[i][th]
      random_var_num += 1
    
  random_acc[th] = random_acc_sum / random_acc_num
  random_var[th] = random_var_sum / random_var_num 

# 推移をプロット
plt.rcParams["font.size"] = 18
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')
# ax1 = fig.add_subplot(2, 2, 1)

ax1.set_ylabel('accuracy')

# ax2 = ax1.twinx()
# ax2.set_ylabel('variance')
 
x = np.array(threshold)
ours = np.array(ours_acc)
top = np.array(top_acc)
random = np.array(random_acc)

ax1.plot(x, ours, color='red', label='ours')
ax1.plot(x, top, color='blue', label='top')
ax1.plot(x, random, color='green', label='random')
ax1.plot(x, x, color='orange', label='threshold', linestyle="dashed")
#ax.plot(left, var_height)
fig.legend(bbox_to_anchor=(0.150, 0.880), loc='upper left')
# plt.savefig("irt-all-pl1.png")
plt.show()

fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')
# ax1 = fig.add_subplot(2, 2, 1)
ax1.set_ylabel('variance')
#ax1.set_ylabel('accuracy')

# ax2 = ax1.twinx()
# ax2.set_ylabel('variance')
 
x = np.array(threshold)
ours = np.array(ours_var)
top = np.array(top_var)
random = np.array(random_var)

ax1.plot(x, ours, color='red', label='ours')
ax1.plot(x, top, color='blue', label='top')
ax1.plot(x, random, color='green', label='random')

#ax.plot(left, var_height)
fig.legend(bbox_to_anchor=(0.150, 0.880), loc='upper left')
# plt.savefig("irt-all-pl1.png")
plt.show()
'''
# 考察: タスク割り当ての実際のパラメータプロット図
plt.rcParams["font.size"] = 18
threshold = [0.75]
sample = devide_sample(task_list, worker_list)
qualify_task = sample['qualify_task']
test_task = sample['test_task']
test_worker = sample['test_worker']

ours_output = make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
# ワーカ候補のリスト
ours_candidate = ours_output[0]
test_worker = ours_output[1]

all_output = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)
irt_candidate = all_output[0]
test_worker = all_output[1]

item_param = all_output[3]
user_param = all_output[4]
#for candidate_dic in ours_candidate.values():
for candidate_dic in irt_candidate.values():
  # print(candidate_dic)
  assign_dic = assignment(candidate_dic, test_worker)

 
item_list = []
worker_list = []
for task in assign_dic:
  item_list.append(item_param[task])
  worker = assign_dic[task]
  worker_list.append(user_param[worker])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(item_list, worker_list)
ax.set_xlabel('task difficulty')
ax.set_ylabel('worker skill')

plt.show()



threshold = list([i / 100 for i in range(60, 81, 10)])


# ワーカー候補だけ作成する
# ワーカーリスト作成~割り当て　実行
ours_all_iter = []
baseline_all_iter = []
iteration_time = 5
for iteration in range(0, iteration_time):
  # results = just_candidate(threshold, label_df, worker_list)
  output =  make_candidate(threshold, input_df, label_df, worker_list, task_list)
  ours_candidate = output[0]

  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker'] 

  # baseline_candidate = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)[0]
  baseline_candidate = entire_top_workers(threshold, input_df, test_worker, qualify_task, test_task)
  # print(results)

  # top-worker 
  # baseline_candidate = entire_top_workers

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
    print(ours_worker_list)   
    for worker_list in baseline_all_iter[iter][th].values():
      for worker in worker_list:
        if worker not in baseline_worker_list:
          baseline_worker_list.append(worker)  

    baseline_num_dic[th] += len(baseline_worker_list)
    ours_num_dic[th] += len(ours_worker_list)
print(ours_num_dic)
print(baseline_num_dic) 
# print(baseline_worker_list)
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
plt.rcParams['font.size'] = 20




# ワーカ能力とタスク困難度の実際の分布
# ヒストグラムを手に入れる
# これはスレッショルド関係ない
iteration_time = 1
for iteration in range(0, iteration_time):
  # results = just_candidate(threshold, label_df, worker_list)
  
  baseline_output = make_candidate_all(threshold, input_df, label_df, worker_list, task_list)
  item_param = list(baseline_output[3].values())
  user_param = list(baseline_output[4].values())
print(item_param)

lim = [-4, 4.5]
worker_map = Frequency_Distribution(user_param, lim, class_width=0.5)
item_map = Frequency_Distribution(item_param, lim, class_width=0.5)
print(worker_map)



bins=np.linspace(-2, 2, 20)
# fig3 = plt.figure()
plt.hist([user_param, item_param], bins, label=['worker', 'task'])
plt.legend(loc='upper left')
plt.xlabel("IRT difficulty")
plt.ylabel("Number of tasks and workers")
# baseline_map.plot.bar(x='階級値', y='度数', label='item', xlabel='difficulty')
plt.show()




