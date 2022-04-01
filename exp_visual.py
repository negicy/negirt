import sys, os
from assignment_method import *
from irt_method import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import girth
import sys
import random
import scipy
import scikit_posthocs as sp
path = os.getcwd()

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
top_acc_allth = []
top_var_allth = []
random_acc_allth = []
random_var_allth = []
full_irt_acc_allth = []
full_irt_var_allth = []

threshold = list([i / 100 for i in range(50, 81)])

# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 40

for iteration in range(0, iteration_time):
  
  ours_acc_perth = []
  ours_var_perth = []

  top_acc_perth = []
  top_var_perth = []

  random_acc_perth = []
  random_var_perth = []

  full_irt_acc_perth = []
  full_irt_var_perth = []

  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker']

  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]

  top_candidate = top_worker_assignment(threshold, input_df, test_worker, qualify_task, test_task)
  full_irt_candidate = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)[0]

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
   
    # 割り当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    top_acc_perth.append(acc)
    top_var_perth.append(var)

  top_acc_allth.append(top_acc_perth)
  top_var_allth.append(top_var_perth)

  for candidate_dic in full_irt_candidate.values():
    assign_dic = assignment(candidate_dic, test_worker)
    # 割当て結果の精度を求める
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker)
  
    full_irt_acc_perth.append(acc)
    full_irt_var_perth.append(var)

  full_irt_acc_allth.append(full_irt_acc_perth)
  full_irt_var_allth.append(full_irt_var_perth)

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
ours_acc_std = []
ours_var_std = []

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)
top_acc_std = []
top_var_std = []

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)
random_acc_std = []
random_var_std = []

full_irt_acc = [0] * len(threshold)
full_irt_var = [0] * len(threshold)
full_acc_std = []
full_var_std = []

for th in range(0, len(threshold)):
  ours_acc_sum = 0
  ours_var_sum = 0
  ours_acc_num = 0
  ours_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if ours_acc_allth[i][th] != "null":
      list_acc_th.append(ours_acc_allth[i][th])
      ours_acc_sum += ours_acc_allth[i][th]
      ours_acc_num += 1
    ours_var_sum += ours_var_allth[i][th]
    list_var_th.append(ours_var_allth[i][th])
    ours_var_num += 1
  # acc, var の平均を計算
  ours_acc[th] = ours_acc_sum / ours_acc_num
  ours_var[th] = ours_var_sum / ours_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  ours_acc_std.append(acc_std)
  ours_var_std.append(var_std)

  if th == 0:
    ours_acc_head = list_acc_th
  if th == len(threshold)-1:
    ours_acc_tail = list_acc_th

  

for th in range(0, len(threshold)):
  top_acc_sum = 0
  top_var_sum = 0
  top_acc_num = 0
  top_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if top_acc_allth[i][th] != "null":
      top_acc_sum += top_acc_allth[i][th]
      list_acc_th.append(top_acc_allth[i][th])
      top_acc_num += 1
    if top_var_allth[i][th] != "null":
      top_var_sum += top_var_allth[i][th]
      list_var_th.append(top_var_allth[i][th])
      top_var_num += 1
    
  top_acc[th] = top_acc_sum / top_acc_num
  top_var[th] = top_var_sum / top_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  top_acc_std.append(acc_std)
  top_var_std.append(var_std)
  if th == 0:
    top_acc_head = list_acc_th
  if th == len(threshold)-1:
    top_acc_tail = list_acc_th
  
for th in range(0, len(threshold)):
  random_acc_sum = 0
  random_var_sum = 0
  random_acc_num = 0
  random_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if random_acc_allth[i][th] != "null":
      random_acc_sum += random_acc_allth[i][th]
      list_acc_th.append(random_acc_allth[i][th])
      random_acc_num += 1
    if random_var_allth[i][th] != "null":
      random_var_sum += random_var_allth[i][th]
      list_var_th.append(random_var_allth[i][th])
      random_var_num += 1
    
  random_acc[th] = random_acc_sum / random_acc_num
  random_var[th] = random_var_sum / random_var_num 
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  random_acc_std.append(acc_std)
  random_var_std.append(var_std)

  if th == 0:
    random_acc_head = list_acc_th
  if th == len(threshold)-1:
    random_acc_tail = list_acc_th

for th in range(0, len(threshold)):
  full_irt_acc_sum = 0
  full_irt_var_sum = 0
  full_irt_acc_num = 0
  full_irt_var_num = 0
  # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
  list_acc_th = []
  list_var_th = []
  for i in range(0, iteration_time):
    #
    if full_irt_acc_allth[i][th] != "null":
      full_irt_acc_sum += full_irt_acc_allth[i][th]
      list_acc_th.append(random_acc_allth[i][th])
      full_irt_acc_num += 1
    if full_irt_var_allth[i][th] != "null":
      full_irt_var_sum += full_irt_var_allth[i][th]
      list_var_th.append(full_irt_var_allth[i][th])
      full_irt_var_num += 1
    
  full_irt_acc[th] = full_irt_acc_sum / full_irt_acc_num
  full_irt_var[th] = full_irt_var_sum / full_irt_var_num
  # 標準偏差を計算
  acc_std = np.std(list_acc_th)
  var_std = np.std(list_var_th)
  full_acc_std.append(acc_std)
  full_var_std.append(var_std)

# 正規性の検定
from scipy import stats as st

print(threshold)
headsData = pd.DataFrame({'top': top_acc_head, 'ours': ours_acc_head, 'random': random_acc_head})
tailData = pd.DataFrame({'top': top_acc_tail, 'ours': ours_acc_tail, 'random': random_acc_tail})
'''
print(headsData)
print(st.bartlett(headsData['top'], headsData['ours'], headsData['random']))
print(st.bartlett(tailData['top'], tailData['ours'], tailData['random']))

print(st.shapiro(headsData))
print(st.shapiro(tailData))

print(st.shapiro(headsData['top']))
print(st.shapiro(headsData['ours']))
print(st.shapiro(headsData['random']))
print(st.shapiro(tailData['top']))
print(st.shapiro(tailData['ours']))
print(st.shapiro(tailData['random']))
'''
#data_head = np.stack( (top_acc_head, ours_acc_head, random_acc_head) )

import scipy.stats as stats

import seaborn as sns
'''
sns.set(font="Hiragino Maru Gothic Pro",context="talk")
fig = plt.subplots(figsize=(8,8))
stats.probplot(top_acc_head, dist="norm", plot=plt)
plt.show()
fig = plt.subplots(figsize=(8,8))
stats.probplot(ours_acc_head, dist="norm", plot=plt)
plt.show()
fig = plt.subplots(figsize=(8,8))
stats.probplot(random_acc_head, dist="norm", plot=plt)
plt.show()
fig = plt.subplots(figsize=(8,8))
stats.probplot(top_acc_tail, dist="norm", plot=plt)
plt.show()
fig = plt.subplots(figsize=(8,8))
stats.probplot(ours_acc_tail, dist="norm", plot=plt)
plt.show()
fig = plt.subplots(figsize=(8,8))
stats.probplot(random_acc_tail, dist="norm", plot=plt)
plt.show()
'''
headsData = pd.DataFrame({'top': top_acc_head, 'ours': ours_acc_head, 'random': random_acc_head})
tailData = pd.DataFrame({'top': top_acc_tail, 'ours': ours_acc_tail, 'random': random_acc_tail})
print(headsData)


#print(scipy.stats.f_oneway(ours_acc_head, top_acc_head))
#print(scipy.stats.f_oneway(ours_acc_head, random_acc_head))

#print(scipy.stats.f_oneway(ours_acc_tail, top_acc_tail))
#print(scipy.stats.f_oneway(ours_acc_tail, random_acc_tail))
'''
anova_head = scipy.stats.f_oneway(ours_acc_head, top_acc_head, random_acc_head)
print(anova_head)
print(scipy.stats.f_oneway(ours_acc_tail, top_acc_tail, random_acc_tail))
'''
print(scipy.stats.kruskal(ours_acc_tail, top_acc_tail, random_acc_tail))
print(scipy.stats.kruskal(ours_acc_head, top_acc_head, random_acc_head))
headsData = headsData.melt(var_name='groups', value_name='values')
tailData = tailData.melt(var_name='groups', value_name='values')
print(sp.posthoc_dscf(headsData, val_col='values', group_col='groups'))
print(sp.posthoc_dscf(tailData, val_col='values', group_col='groups'))


#print(pairwise(headsData.))



# 推移をプロット
plt.rcParams["font.size"] = 18
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')


ax1.set_ylabel('accuracy')
x = np.array(threshold)
ours_acc = np.array(ours_acc)
top_acc = np.array(top_acc)
random_acc = np.array(random_acc)
full_irt_acc = np.array(full_irt_acc)

ax1.plot(x, ours_acc, color='red', label='ours')
ax1.plot(x, top_acc, color='blue', label='top')
ax1.plot(x, random_acc, color='green', label='random')
# ax1.plot(x, full, color='purple', label='IRT')
ax1.plot(x, x, color='orange', linestyle="dashed")
# ax.plot(left, var_height)

# 正解率 ours-random-top
a = 0.15
plt.fill_between(x, ours_acc - ours_acc_std, ours_acc+ours_acc_std, facecolor='r', alpha=a)
plt.fill_between(x, top_acc - top_acc_std, top_acc + top_acc_std, facecolor='b', alpha=a)
plt.fill_between(x, random_acc - random_acc_std, random_acc + random_acc_std, facecolor='g', alpha=a)
fig.legend(bbox_to_anchor=(0.280, 0.400), loc='upper left')
plt.show()

# 正解率 ours-irt
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')

ax1.set_ylabel('accuracy')
ax1.plot(x, x, color='orange', label='threshold', linestyle="dashed")
ax1.plot(x, ours_acc, color='red', label='ours')
ax1.plot(x, full_irt_acc, color='purple', label='IRT')
plt.fill_between(x, ours_acc - ours_acc_std, ours_acc+ours_acc_std, facecolor='r', alpha=a)
plt.fill_between(x, full_irt_acc - full_acc_std, full_irt_acc + full_acc_std, facecolor='purple', alpha=a)
fig.legend(bbox_to_anchor=(0.150, 0.380), loc='upper left')
plt.show()


# 分散 ours-random-top
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')
ax1.set_ylabel('variance')

ours_var = np.array(ours_var)
top_var = np.array(top_var)
random_var = np.array(random_var)
full_irt_var = np.array(full_irt_var)

ax1.plot(x, ours_var, color='red', label='ours')
ax1.plot(x, top_var, color='blue', label='top')
ax1.plot(x, random_var, color='green', label='random')

plt.fill_between(x, ours_var - ours_var_std, ours_var + ours_var_std, facecolor='r', alpha=a)
# plt.fill_between(x, full_irt_var - full_var_std, full_irt_var + full_var_std, facecolor='r', alpha=a)
plt.fill_between(x, top_var - top_var_std, top_var + top_var_std, facecolor='b', alpha=a)
plt.fill_between(x, random_var - random_var_std, random_var + random_var_std, facecolor='g', alpha=a)
fig.legend(bbox_to_anchor=(0.150, 0.580), loc='upper left')
plt.show()

# 分散 ours-irt
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('threshold')
# ax1 = fig.add_subplot(2, 2, 1)
ax1.set_ylabel('variance')
ax1.plot(x, ours_var, color='red', label='ours')
ax1.plot(x, full_irt_var, color='purple', label='ours')

plt.fill_between(x, ours_var - ours_var_std, ours_var + ours_var_std, facecolor='r', alpha=a)
plt.fill_between(x, full_irt_var - full_var_std, full_irt_var + full_var_std, facecolor='purple', alpha=a)
fig.legend(bbox_to_anchor=(0.150, 0.880), loc='upper left')

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

all_output = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)
irt_candidate = all_output[0]
item_param = all_output[1]
user_param = all_output[2]
for candidate_dic in ours_candidate.values():
  ours_assign_dic = assignment(candidate_dic, test_worker)

for candidate_dic in irt_candidate.values():
  # print(candidate_dic)
  all_assign_dic = assignment(candidate_dic, test_worker)

print(item_param)
ours_item_list = []
ours_user_list = []

all_item_list = []
all_user_list = []
# タスクとワーカの実際のパラメータを格納
for task in all_assign_dic:
  ours_item_list.append(item_param[task])
  worker = ours_assign_dic[task]
  ours_user_list.append(user_param[worker])

for task in all_assign_dic:
  
  all_item_list.append(item_param[task])
  worker = all_assign_dic[task]
  all_user_list.append(user_param[worker])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
print(len(all_item_list))
print(len(ours_item_list))
#ax.scatter(ours_item_list, ours_user_list, marker="^", label="ours")
ax.scatter(all_item_list, all_user_list, label="Assignment")
plt.legend()
ax.set_xlabel('task difficulty')
ax.set_ylabel('worker skill')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#ax.scatter(ours_item_list, ours_user_list, marker="^", label="ours")
ax.scatter(ours_item_list, ours_user_list, label="Assignment")
plt.legend()
ax.set_xlabel('task difficulty')
ax.set_ylabel('worker skill')

plt.show()


# ワーカ候補作成ヒストグラム
threshold = list([i / 100 for i in range(60, 81, 10)])


# ワーカー候補だけ作成する
# ワーカーリスト作成~割り当て　実行
ours_all_iter = []
baseline_all_iter = []
iteration_time = 5
for iteration in range(0, iteration_time):
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker'] 

  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]

  baseline_candidate = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, qualify_task, test_task)[0]
  # baseline_candidate = entire_top_workers(threshold, input_df, test_worker, qualify_task, test_task)
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

'''

'''
# ワーカ能力とタスク困難度の実際の分布
# ヒストグラムを手に入れる
# これはスレッショルド関係ない
# タスク100件, ワーカ全員
# worker_list = list(input_df.columns)
iteration_time = 1
for iteration in range(0, iteration_time):
  print(worker_list)
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker'] 
  
  baseline_output = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)
  item_param = list(baseline_output[1].values())
  user_param = list(baseline_output[2].values())
print(item_param)


lim = [-4, 4.5]
worker_map = Frequency_Distribution(user_param, lim, class_width=0.5)
item_map = Frequency_Distribution(item_param, lim, class_width=0.5)
print(worker_map)
print(item_map)



bins=np.linspace(-4, 4, 20)
# fig3 = plt.figure()
plt.hist([user_param, item_param], bins, label=['worker', 'task'])
plt.legend(loc='upper left')
plt.xlabel("IRT parameter")
plt.ylabel("Number of tasks and workers")
# baseline_map.plot.bar(x='階級値', y='度数', label='item', xlabel='difficulty')
# plt.show()

'''


